"""
3D Gaussian Splatting Trainer using SlangPy
"""

import slangpy as spy
import numpy as np
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple
import json


@dataclass
class TrainingConfig:
    """Configuration for 3DGS training"""
    max_splats: int = 5_000_000
    max_tiles: int = 14400  # (1920/16)^2
    tile_size: int = 16
    max_gaussians_per_tile: int = 1024
    
    # Training params
    num_views_per_batch: int = 6
    start_densification_iter: int = 500
    end_densification_iter: int = 15000
    
    # Gradient descent flags
    descend_mean: bool = True
    descend_scale: bool = True
    descend_rotation: bool = True
    descend_color: bool = True
    descend_opacity: bool = True
    enable_densification: bool = True
    
    # Initial gaussian params
    initial_scale: float = 0.01
    initial_opacity: float = 0.5


class GaussianSplattingTrainer:
    """
    3DGS Training Pipeline using SlangPy
    
    Manages GPU buffers and dispatches compute kernels for:
    - Binning gaussians into tiles
    - Sorting per tile by depth
    - Forward rasterization
    - Backward gradient computation
    - Adam optimizer update
    - Densification and compaction
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        render_width: int,
        render_height: int,
        num_initial_gaussians: int = 100_000
    ):
        self.config = config
        self.render_width = render_width
        self.render_height = render_height
        self.num_gaussians = num_initial_gaussians
        self.iteration = 0
        
        # Calculate tile counts
        self.num_tiles_x = (render_width + config.tile_size - 1) // config.tile_size
        self.num_tiles_y = (render_height + config.tile_size - 1) // config.tile_size
        self.total_tiles = self.num_tiles_x * self.num_tiles_y
        
        # Create SlangPy device
        shader_path = pathlib.Path(__file__).parent / "shaders"
        self.device = spy.create_device(include_paths=[shader_path])
        
        # Load shader module (for struct types)
        self.module = spy.Module.load_from_file(
            self.device,
            str(shader_path / "kernels.slang")
        )
        
        # Create compute kernels for each entry point
        self._create_kernels(shader_path)
        
        # Allocate GPU buffers
        self._allocate_buffers()
        
        # Camera and target image data (will be set by load_colmap_data)
        self.cameras_buffer = None
        self.target_images_texture = None
        self.num_cameras = 0
        
        print(f"Initialized 3DGS Trainer:")
        print(f"  Render size: {render_width}x{render_height}")
        print(f"  Tiles: {self.num_tiles_x}x{self.num_tiles_y} = {self.total_tiles}")
        print(f"  Max splats: {config.max_splats:,}")
        print(f"  Initial gaussians: {num_initial_gaussians:,}")
    
    def _create_kernels(self, shader_path: pathlib.Path):
        """Create ComputeKernel objects for each shader entry point"""
        kernel_path = str(shader_path / "kernels.slang")
        
        def create_kernel(entry_point: str):
            program = self.device.load_program(kernel_path, [entry_point])
            desc = spy.ComputeKernelDesc()
            desc.program = program
            return self.device.create_compute_kernel(desc)
        
        # Create all compute kernels
        self.kernel_clearTileBins = create_kernel("clearTileBins")
        self.kernel_clearProjectedGaussians = create_kernel("clearProjectedGaussians")
        self.kernel_clearTextures = create_kernel("clearTextures")
        self.kernel_binGaussians = create_kernel("binGaussians")
        self.kernel_sortGaussiansPerTile = create_kernel("sortGaussiansPerTile")
        self.kernel_forwardRasterize = create_kernel("forwardRasterize")
        self.kernel_backwardRasterize = create_kernel("backwardRasterize")
        self.kernel_updateGaussiansAdam = create_kernel("updateGaussiansAdam")
        self.kernel_clearGradientBuffers = create_kernel("clearGradientBuffers")
        self.kernel_clearAdamState = create_kernel("clearAdamState")
        self.kernel_initializeGaussians = create_kernel("initializeGaussians")
    
    def _allocate_buffers(self):
        """Allocate all persistent GPU buffers"""
        cfg = self.config
        
        # === Gaussian data buffers ===
        self.gaussians = spy.NDBuffer(
            self.device,
            dtype=self.module.Struct_Gaussian3D,
            shape=(cfg.max_splats,)
        )
        
        self.projected_gaussians_2d = spy.NDBuffer(
            self.device,
            dtype=self.module.Struct_Projected2DGaussian,
            shape=(cfg.max_splats,)
        )
        
        self.adam_state = spy.NDBuffer(
            self.device,
            dtype=self.module.Struct_AdamState,
            shape=(cfg.max_splats,)
        )
        
        # === Tile binning buffers ===
        self.screen_tiles_count = spy.NDBuffer(
            self.device,
            dtype=spy.uint1,
            shape=(cfg.max_tiles,)
        )
        
        # Combined buffer for tile indices (used for binning output and sorted IDs)
        self.screen_tiles_indices = spy.NDBuffer(
            self.device,
            dtype=spy.uint1,
            shape=(cfg.max_tiles * cfg.max_gaussians_per_tile,)
        )
        
        # === Gradient accumulator buffers (14 separate int32 buffers) ===
        self.mean_gradient_accum_x = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        self.mean_gradient_accum_y = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        self.mean_gradient_accum_z = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        self.mean2d_gradient_accum_magnitude = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        
        self.scale_gradient_accum_x = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        self.scale_gradient_accum_y = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        self.scale_gradient_accum_z = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        
        self.rotation_gradient_accum_x = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        self.rotation_gradient_accum_y = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        self.rotation_gradient_accum_z = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        self.rotation_gradient_accum_w = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        
        self.color_gradient_accum_r = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        self.color_gradient_accum_g = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        self.color_gradient_accum_b = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        
        self.opacity_gradient_accum = spy.NDBuffer(self.device, dtype=spy.int1, shape=(cfg.max_splats,))
        
        # === Auxiliary buffers ===
        self.pixel_coverage_count_per_gaussian = spy.NDBuffer(
            self.device,
            dtype=spy.uint1,
            shape=(cfg.max_splats,)
        )
        
        self.num_gaussians_counter = spy.NDBuffer(
            self.device,
            dtype=spy.uint1,
            shape=(1,)
        )
        
        self.training_step_counter = spy.NDBuffer(
            self.device,
            dtype=spy.uint1,
            shape=(1,)
        )
        
        # === Output textures ===
        self.output_image = self.device.create_texture(
            width=self.render_width,
            height=self.render_height,
            format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
        )
        
        self.output_transmittance = self.device.create_texture(
            width=self.render_width,
            height=self.render_height,
            format=spy.Format.r32_float,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
        )
        
        self.gaussian_count_per_pixel = self.device.create_texture(
            width=self.render_width,
            height=self.render_height,
            format=spy.Format.r32_uint,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
        )
        
        # Create sampler for target images
        self.target_sampler = self.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge
        )
    
    def load_colmap_data(self, colmap_path: str):
        """
        Load COLMAP data: cameras and images
        
        Args:
            colmap_path: Path to COLMAP output folder containing cameras.bin/txt and images folder
        """
        from colmap_loader import load_colmap_cameras, load_colmap_images
        
        colmap_path = pathlib.Path(colmap_path)
        
        # Load camera intrinsics/extrinsics
        cameras, images = load_colmap_cameras(colmap_path)
        self.num_cameras = len(images)
        
        # Create CameraData buffer
        # Build camera data as numpy structured array first
        # (cursor.write() doesn't flush to GPU properly)
        cam_dtype = np.dtype([
            ('view_matrix', np.float32, (4, 4)),
            ('projection_matrix', np.float32, (4, 4)),
            ('image_size', np.float32, 2),
            ('image_index', np.uint32),
            ('_pad', np.float32)
        ])
        
        camera_data = np.zeros(self.num_cameras, dtype=cam_dtype)
        
        for i, img in enumerate(images):
            cam = cameras[img['camera_id']]
            # View matrix needs to be transposed for shader convention
            # (matching prepare_colmap_iteration_data.slang)
            camera_data[i]['view_matrix'] = img['view_matrix'].T
            camera_data[i]['projection_matrix'] = cam['projection_matrix']
            camera_data[i]['image_size'] = [cam['width'], cam['height']]
            camera_data[i]['image_index'] = img['index']
        
        self.cameras_buffer = spy.NDBuffer(
            self.device,
            dtype=self.module.CameraData,
            shape=(self.num_cameras,)
        )
        self.cameras_buffer.copy_from_numpy(camera_data.view(np.uint8))
        
        # Load images into texture array
        image_arrays = load_colmap_images(colmap_path, images, self.render_width, self.render_height)
        
        self.target_images_texture = self.device.create_texture(
            type=spy.TextureType.texture_2d_array,
            width=self.render_width,
            height=self.render_height,
            array_length=self.num_cameras,
            format=spy.Format.rgba8_unorm,
            usage=spy.TextureUsage.shader_resource
        )
        
        # Upload all images to texture array
        for i, img_array in enumerate(image_arrays):
            self.target_images_texture.copy_from_numpy(img_array, layer=i)
        
        print(f"Loaded {self.num_cameras} cameras/images from COLMAP")
    
    def initialize_gaussians(
        self,
        positions: Optional[np.ndarray] = None,
        scene_min: Tuple[float, float, float] = (-1, -1, -1),
        scene_max: Tuple[float, float, float] = (1, 1, 1)
    ):
        """
        Initialize gaussian positions
        
        Args:
            positions: Optional (N, 3) array of initial positions. If None, random positions are used.
            scene_min: Minimum bounds for random initialization
            scene_max: Maximum bounds for random initialization
        """
        if positions is not None:
            # Initialize from provided positions (e.g., COLMAP point cloud)
            self.num_gaussians = min(len(positions), self.config.max_splats)
            
            # Build gaussian data as numpy structured array first
            # (cursor.write() doesn't flush to GPU properly)
            gaussian_dtype = np.dtype([
                ('mean', np.float32, 3),
                ('log_scale', np.float32, 3),
                ('rotation_quat', np.float32, 4),
                ('opacity', np.float32),
                ('color_logit', np.float32, 3)
            ])
            
            gaussian_data = np.zeros(self.num_gaussians, dtype=gaussian_dtype)
            log_scale = np.log(self.config.initial_scale)
            
            for i in range(self.num_gaussians):
                gaussian_data[i]['mean'] = positions[i]
                gaussian_data[i]['log_scale'] = [log_scale, log_scale, log_scale]
                gaussian_data[i]['rotation_quat'] = [1, 0, 0, 0]  # Identity (w, x, y, z)
                gaussian_data[i]['opacity'] = self.config.initial_opacity
                gaussian_data[i]['color_logit'] = [0, 0, 0]  # Mid-gray
            
            # Copy to GPU buffer
            self.gaussians.copy_from_numpy(gaussian_data.view(np.uint8))
            
            # Initialize Adam state to zeros
            groups = (self.config.max_splats + 255) // 256
            self.kernel_clearAdamState.dispatch(
                thread_count=[groups, 1, 1],
                max_splats=self.config.max_splats,
                adam_state=self.adam_state.storage
            )
            
            # Set counter
            self.num_gaussians_counter.copy_from_numpy(np.array([self.num_gaussians], dtype=np.uint32))
        else:
            # Use GPU kernel for random initialization
            groups = (self.num_gaussians + 255) // 256
            self.kernel_initializeGaussians.dispatch(
                thread_count=[groups, 1, 1],
                num_gaussians=self.num_gaussians,
                scene_min=spy.float3(scene_min[0], scene_min[1], scene_min[2]),
                scene_max=spy.float3(scene_max[0], scene_max[1], scene_max[2]),
                initial_scale=self.config.initial_scale,
                initial_opacity=self.config.initial_opacity,
                input_gaussians=self.gaussians.storage,
                adam_state=self.adam_state.storage,
                num_gaussians_counter=self.num_gaussians_counter.storage
            )
        
        # Clear gradient buffers
        self._clear_gradient_buffers()
        
        print(f"Initialized {self.num_gaussians:,} gaussians")
    
    def _clear_gradient_buffers(self):
        """Clear all gradient accumulator buffers"""
        groups = (self.config.max_splats + 255) // 256
        self.kernel_clearGradientBuffers.dispatch(
            thread_count=[groups, 1, 1],
            max_splats=self.config.max_splats,
            mean_gradient_accum_x=self.mean_gradient_accum_x.storage,
            mean_gradient_accum_y=self.mean_gradient_accum_y.storage,
            mean_gradient_accum_z=self.mean_gradient_accum_z.storage,
            mean2d_gradient_accum_magnitude=self.mean2d_gradient_accum_magnitude.storage,
            scale_gradient_accum_x=self.scale_gradient_accum_x.storage,
            scale_gradient_accum_y=self.scale_gradient_accum_y.storage,
            scale_gradient_accum_z=self.scale_gradient_accum_z.storage,
            rotation_gradient_accum_x=self.rotation_gradient_accum_x.storage,
            rotation_gradient_accum_y=self.rotation_gradient_accum_y.storage,
            rotation_gradient_accum_z=self.rotation_gradient_accum_z.storage,
            rotation_gradient_accum_w=self.rotation_gradient_accum_w.storage,
            color_gradient_accum_r=self.color_gradient_accum_r.storage,
            color_gradient_accum_g=self.color_gradient_accum_g.storage,
            color_gradient_accum_b=self.color_gradient_accum_b.storage,
            opacity_gradient_accum=self.opacity_gradient_accum.storage
        )
    
    def forward_pass(self, camera_index: int):
        """
        Execute forward rendering pass
        
        Args:
            camera_index: Index of camera/target image to render
        """
        frame_dim = spy.float2(float(self.render_width), float(self.render_height))
        
        # Clear tile bins
        tile_groups = (self.total_tiles + 255) // 256
        self.kernel_clearTileBins.dispatch(
            thread_count=[tile_groups, 1, 1],
            total_tiles=self.total_tiles,
            screen_tiles_count=self.screen_tiles_count.storage,
            screen_tiles_indices=self.screen_tiles_indices.storage
        )
        
        # Clear projected gaussians
        gaussian_groups = (self.config.max_splats + 255) // 256
        self.kernel_clearProjectedGaussians.dispatch(
            thread_count=[gaussian_groups, 1, 1],
            max_splats=self.config.max_splats,
            projected_gaussians_2d=self.projected_gaussians_2d.storage
        )
        
        # Clear output textures
        pixel_groups_x = (self.render_width + 15) // 16
        pixel_groups_y = (self.render_height + 15) // 16
        self.kernel_clearTextures.dispatch(
            thread_count=[pixel_groups_x, pixel_groups_y, 1],
            frame_dim=frame_dim,
            output_image=self.output_image,
            output_transmittance=self.output_transmittance,
            gaussian_count_per_pixel=self.gaussian_count_per_pixel
        )
        
        # Bin gaussians into tiles
        bin_groups = (self.num_gaussians + 255) // 256
        self.kernel_binGaussians.dispatch(
            thread_count=[bin_groups, 1, 1],
            frame_dim=frame_dim,
            num_gaussians=self.num_gaussians,
            input_gaussians=self.gaussians.storage,
            cameras=self.cameras_buffer.storage,
            camera_index=camera_index,
            projected_gaussians_2d=self.projected_gaussians_2d.storage,
            screen_tiles_count=self.screen_tiles_count.storage,
            screen_tiles_indices=self.screen_tiles_indices.storage,
            training_step_counter=self.training_step_counter.storage
        )
        
        # Sort gaussians per tile by depth
        self.kernel_sortGaussiansPerTile.dispatch(
            thread_count=[tile_groups, 1, 1],
            frame_dim=frame_dim,
            screen_tiles_count=self.screen_tiles_count.storage,
            screen_tiles_indices=self.screen_tiles_indices.storage,
            projected_gaussians_2d=self.projected_gaussians_2d.storage
        )
        
        # Forward rasterize
        self.kernel_forwardRasterize.dispatch(
            thread_count=[pixel_groups_x, pixel_groups_y, 1],
            frame_dim=frame_dim,
            input_gaussians=self.gaussians.storage,
            projected_gaussians_2d=self.projected_gaussians_2d.storage,
            screen_tiles_count=self.screen_tiles_count.storage,
            screen_tiles_indices=self.screen_tiles_indices.storage,
            output_image=self.output_image,
            output_transmittance=self.output_transmittance,
            gaussian_count_per_pixel=self.gaussian_count_per_pixel,
            pixel_coverage_count_per_gaussian=self.pixel_coverage_count_per_gaussian.storage
        )
    
    def backward_pass(self, camera_index: int):
        """
        Execute backward gradient computation pass
        
        Args:
            camera_index: Index of camera/target image for loss computation
        """
        frame_dim = spy.float2(float(self.render_width), float(self.render_height))
        cfg = self.config
        
        pixel_groups_x = (self.render_width + 15) // 16
        pixel_groups_y = (self.render_height + 15) // 16
        
        self.kernel_backwardRasterize.dispatch(
            thread_count=[pixel_groups_x, pixel_groups_y, 1],
            frame_dim=frame_dim,
            descend_mean=cfg.descend_mean,
            descend_scale=cfg.descend_scale,
            descend_rotation=cfg.descend_rotation,
            descend_color=cfg.descend_color,
            descend_opacity=cfg.descend_opacity,
            input_gaussians=self.gaussians.storage,
            projected_gaussians_2d=self.projected_gaussians_2d.storage,
            cameras=self.cameras_buffer.storage,
            camera_index=camera_index,
            target_images=self.target_images_texture,
            target_sampler=self.target_sampler,
            output_image=self.output_image,
            output_transmittance=self.output_transmittance,
            screen_tiles_count=self.screen_tiles_count.storage,
            screen_tiles_indices=self.screen_tiles_indices.storage,
            gaussian_count_per_pixel=self.gaussian_count_per_pixel,
            mean_gradient_accum_x=self.mean_gradient_accum_x.storage,
            mean_gradient_accum_y=self.mean_gradient_accum_y.storage,
            mean_gradient_accum_z=self.mean_gradient_accum_z.storage,
            mean2d_gradient_accum_magnitude=self.mean2d_gradient_accum_magnitude.storage,
            scale_gradient_accum_x=self.scale_gradient_accum_x.storage,
            scale_gradient_accum_y=self.scale_gradient_accum_y.storage,
            scale_gradient_accum_z=self.scale_gradient_accum_z.storage,
            rotation_gradient_accum_x=self.rotation_gradient_accum_x.storage,
            rotation_gradient_accum_y=self.rotation_gradient_accum_y.storage,
            rotation_gradient_accum_z=self.rotation_gradient_accum_z.storage,
            rotation_gradient_accum_w=self.rotation_gradient_accum_w.storage,
            color_gradient_accum_r=self.color_gradient_accum_r.storage,
            color_gradient_accum_g=self.color_gradient_accum_g.storage,
            color_gradient_accum_b=self.color_gradient_accum_b.storage,
            opacity_gradient_accum=self.opacity_gradient_accum.storage
        )
    
    def optimizer_step(self):
        """Execute Adam optimizer update"""
        cfg = self.config
        
        groups = (self.num_gaussians + 255) // 256
        self.kernel_updateGaussiansAdam.dispatch(
            thread_count=[groups, 1, 1],
            num_gaussians=self.num_gaussians,
            num_views=cfg.num_views_per_batch,
            iteration=self.iteration,
            descend_mean=cfg.descend_mean,
            descend_scale=cfg.descend_scale,
            descend_rotation=cfg.descend_rotation,
            descend_color=cfg.descend_color,
            descend_opacity=cfg.descend_opacity,
            enable_densification=cfg.enable_densification,
            start_densification_iter=cfg.start_densification_iter,
            end_densification_iter=cfg.end_densification_iter,
            max_splats=cfg.max_splats,
            input_gaussians=self.gaussians.storage,
            adam_state=self.adam_state.storage,
            num_gaussians_counter=self.num_gaussians_counter.storage,
            pixel_coverage_count_per_gaussian=self.pixel_coverage_count_per_gaussian.storage,
            mean_gradient_accum_x=self.mean_gradient_accum_x.storage,
            mean_gradient_accum_y=self.mean_gradient_accum_y.storage,
            mean_gradient_accum_z=self.mean_gradient_accum_z.storage,
            mean2d_gradient_accum_magnitude=self.mean2d_gradient_accum_magnitude.storage,
            scale_gradient_accum_x=self.scale_gradient_accum_x.storage,
            scale_gradient_accum_y=self.scale_gradient_accum_y.storage,
            scale_gradient_accum_z=self.scale_gradient_accum_z.storage,
            rotation_gradient_accum_x=self.rotation_gradient_accum_x.storage,
            rotation_gradient_accum_y=self.rotation_gradient_accum_y.storage,
            rotation_gradient_accum_z=self.rotation_gradient_accum_z.storage,
            rotation_gradient_accum_w=self.rotation_gradient_accum_w.storage,
            color_gradient_accum_r=self.color_gradient_accum_r.storage,
            color_gradient_accum_g=self.color_gradient_accum_g.storage,
            color_gradient_accum_b=self.color_gradient_accum_b.storage,
            opacity_gradient_accum=self.opacity_gradient_accum.storage
        )
    
    def train_step(self, camera_index: int) -> np.ndarray:
        """
        Execute one full training step: forward, backward, and optimizer update
        
        Args:
            camera_index: Index of camera/target image to train on
            
        Returns:
            Rendered image as numpy array (H, W, 4)
        """
        # Forward pass
        self.forward_pass(camera_index)
        
        # Backward pass
        self.backward_pass(camera_index)
        
        # Optimizer step (only updates after num_views_per_batch iterations)
        self.optimizer_step()
        
        self.iteration += 1
        
        # Return rendered image
        return self.get_rendered_image()
    
    def get_rendered_image(self) -> np.ndarray:
        """Get the current rendered image as numpy array"""
        return self.output_image.to_numpy()
    
    def get_num_gaussians(self) -> int:
        """Get current number of active gaussians"""
        return int(self.num_gaussians_counter.to_numpy().flatten()[0])
    
    def train(
        self,
        num_iterations: int,
        log_interval: int = 100,
        save_interval: int = 1000,
        image_interval: int = 250,
        output_path: Optional[str] = None
    ):
        """
        Run full training loop
        
        Args:
            num_iterations: Total number of iterations
            log_interval: Iterations between logging
            save_interval: Iterations between saving checkpoints
            image_interval: Iterations between saving preview images
            output_path: Path to save outputs
        """
        import time
        from PIL import Image
        
        if self.num_cameras == 0:
            raise RuntimeError("No camera data loaded. Call load_colmap_data() first.")
        
        print(f"Starting training for {num_iterations} iterations...")
        start_time = time.time()
        
        for i in range(num_iterations):
            # Random camera selection
            camera_index = np.random.randint(0, self.num_cameras)
            
            # Train step
            rendered = self.train_step(camera_index)
            
            # Logging
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                its_per_sec = (i + 1) / elapsed
                num_g = self.get_num_gaussians()
                print(f"Iter {i+1}/{num_iterations} | {its_per_sec:.1f} it/s | Gaussians: {num_g:,}")
            
            # Save preview image
            if output_path and (i + 1) % image_interval == 0:
                # Debug: check tile counts
                tile_counts = self.screen_tiles_count.to_numpy().flatten()
                print(f"  Tiles with gaussians: {(tile_counts > 0).sum()}, max count: {tile_counts.max()}")
                self.save_preview_image(f"{output_path}/preview_{i+1:06d}.png")
            
            # Save checkpoint
            if output_path and (i + 1) % save_interval == 0:
                self.save_checkpoint(f"{output_path}/checkpoint_{i+1}.npz")
        
        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.1f}s ({num_iterations/total_time:.1f} it/s)")
    
    def save_preview_image(self, path: str):
        """Save current rendered image to file"""
        from PIL import Image
        
        # Get rendered image (H, W, 4) float32
        img_float = self.get_rendered_image()
        
        # Debug: check image stats
        print(f"  Preview: shape={img_float.shape}, min={img_float.min():.4f}, max={img_float.max():.4f}")
        
        # Clamp and convert to uint8
        img_uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        
        # Save as PNG
        Image.fromarray(img_uint8).save(path)
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        gaussians_np = self.gaussians.to_numpy()[:self.get_num_gaussians()]
        adam_np = self.adam_state.to_numpy()[:self.get_num_gaussians()]
        
        np.savez(
            path,
            gaussians=gaussians_np,
            adam_state=adam_np,
            iteration=self.iteration,
            num_gaussians=self.get_num_gaussians()
        )
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        data = np.load(path, allow_pickle=True)
        
        self.num_gaussians = int(data['num_gaussians'])
        self.iteration = int(data['iteration'])
        
        self.gaussians.copy_from_numpy(data['gaussians'])
        self.adam_state.copy_from_numpy(data['adam_state'])
        self.num_gaussians_counter.copy_from_numpy(np.array([self.num_gaussians], dtype=np.uint32))
        
        print(f"Loaded checkpoint from {path} (iter {self.iteration}, {self.num_gaussians:,} gaussians)")


def main():
    """Example training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="3DGS Training with SlangPy")
    parser.add_argument("--colmap_path", type=str, required=True, help="Path to COLMAP output")
    parser.add_argument("--output_path", type=str, default="./output", help="Output directory")
    parser.add_argument("--iterations", type=int, default=30000, help="Training iterations")
    parser.add_argument("--width", type=int, default=800, help="Render width")
    parser.add_argument("--height", type=int, default=800, help="Render height")
    parser.add_argument("--initial_gaussians", type=int, default=100000, help="Initial gaussian count")
    args = parser.parse_args()
    
    # Create output directory
    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    config = TrainingConfig()
    trainer = GaussianSplattingTrainer(
        config=config,
        render_width=args.width,
        render_height=args.height,
        num_initial_gaussians=args.initial_gaussians
    )
    
    # Load COLMAP data
    trainer.load_colmap_data(args.colmap_path)
    
    # Load COLMAP point cloud and use for initialization
    from colmap_loader import load_colmap_points
    points = load_colmap_points(pathlib.Path(args.colmap_path))
    
    if len(points) > 0:
        print(f"Using {len(points)} COLMAP points for initialization")
        trainer.initialize_gaussians(positions=points)
    else:
        # Fallback to random initialization with computed bounds
        from colmap_loader import load_colmap_cameras
        cameras, images = load_colmap_cameras(pathlib.Path(args.colmap_path))
        cam_positions = []
        for img in images:
            view_mat = img['view_matrix']
            R = view_mat[:3, :3]
            t = view_mat[:3, 3]
            cam_pos = -R.T @ t
            cam_positions.append(cam_pos)
        cam_positions = np.array(cam_positions)
        
        cam_center = cam_positions.mean(axis=0)
        cam_dist = np.linalg.norm(cam_positions - cam_center, axis=1).mean()
        scene_radius = cam_dist * 0.3
        scene_center = np.array([0, 0, 0])
        
        scene_min = tuple(scene_center - scene_radius)
        scene_max = tuple(scene_center + scene_radius)
        print(f"No COLMAP points found, using random init with bounds: {scene_min} to {scene_max}")
        trainer.initialize_gaussians(scene_min=scene_min, scene_max=scene_max)
    
    # Train
    trainer.train(
        num_iterations=args.iterations,
        log_interval=100,
        save_interval=5000,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
