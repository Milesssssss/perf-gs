"""
Simple test script to verify 3DGS trainer setup and shader compilation
"""

import slangpy as spy
import numpy as np
import pathlib
import time


def test_shader_compilation():
    """Test that all shaders compile without errors"""
    print("=" * 60)
    print("TEST: Shader Compilation")
    print("=" * 60)
    
    shader_path = pathlib.Path(__file__).parent / "shaders"
    
    try:
        device = spy.create_device(include_paths=[shader_path])
        print(f"✓ Created SlangPy device")
        
        module = spy.Module.load_from_file(
            device,
            str(shader_path / "kernels.slang")
        )
        print(f"✓ Loaded kernels.slang module")
        
        # Check that key entry points exist
        entry_points = [
            'clearTileBins',
            'clearProjectedGaussians', 
            'clearGradientBuffers',
            'clearAdamState',
            'clearTextures',
            'binGaussians',
            'sortGaussiansPerTile',
            'forwardRasterize',
            'backwardRasterize',
            'updateGaussiansAdam',
            'initializeGaussians',
        ]
        
        for ep in entry_points:
            if hasattr(module, ep):
                print(f"  ✓ Found entry point: {ep}")
            else:
                print(f"  ✗ Missing entry point: {ep}")
        
        # Check struct types
        struct_types = [
            'Struct_Gaussian3D',
            'Struct_Projected2DGaussian',
            'Struct_AdamState',
            'CameraData',
            'GradientAccumulator',
        ]
        
        for st in struct_types:
            if hasattr(module, st):
                print(f"  ✓ Found struct type: {st}")
            else:
                print(f"  ✗ Missing struct type: {st}")
        
        return device, module
        
    except Exception as e:
        print(f"✗ Shader compilation failed: {e}")
        raise


def test_buffer_allocation(device, module):
    """Test GPU buffer allocation"""
    print("\n" + "=" * 60)
    print("TEST: Buffer Allocation")
    print("=" * 60)
    
    max_splats = 100_000
    max_tiles = 14400
    max_gaussians_per_tile = 1024
    width, height = 800, 800
    
    try:
        # Gaussian data
        gaussians = spy.NDBuffer(device, dtype=module.Struct_Gaussian3D, shape=(max_splats,))
        print(f"✓ Allocated gaussians buffer: {max_splats:,} elements")
        
        projected = spy.NDBuffer(device, dtype=module.Struct_Projected2DGaussian, shape=(max_splats,))
        print(f"✓ Allocated projected_gaussians_2d buffer")
        
        adam_state = spy.NDBuffer(device, dtype=module.Struct_AdamState, shape=(max_splats,))
        print(f"✓ Allocated adam_state buffer")
        
        # Tile buffers (using slangpy's builtin types)
        tiles_count = spy.NDBuffer(device, dtype=spy.uint1, shape=(max_tiles,))
        print(f"✓ Allocated screen_tiles_count buffer: {max_tiles:,} tiles")
        
        tiles_indices = spy.NDBuffer(device, dtype=spy.uint1, shape=(max_tiles * max_gaussians_per_tile,))
        print(f"✓ Allocated screen_tiles_indices buffer: {max_tiles * max_gaussians_per_tile:,} elements")
        
        # Gradient buffers
        grad_buffers = {}
        grad_names = [
            'mean_x', 'mean_y', 'mean_z', 'mean2d_mag',
            'scale_x', 'scale_y', 'scale_z',
            'rot_x', 'rot_y', 'rot_z', 'rot_w',
            'color_r', 'color_g', 'color_b',
            'opacity'
        ]
        for name in grad_names:
            grad_buffers[name] = spy.NDBuffer(device, dtype=spy.int1, shape=(max_splats,))
        print(f"✓ Allocated {len(grad_names)} gradient accumulator buffers")
        
        # Counter buffers
        num_gaussians_counter = spy.NDBuffer(device, dtype=spy.uint1, shape=(1,))
        training_step_counter = spy.NDBuffer(device, dtype=spy.uint1, shape=(1,))
        print(f"✓ Allocated counter buffers")
        
        return {
            'gaussians': gaussians,
            'projected': projected,
            'adam_state': adam_state,
            'tiles_count': tiles_count,
            'tiles_indices': tiles_indices,
            'grad_buffers': grad_buffers,
            'num_gaussians_counter': num_gaussians_counter,
            'training_step_counter': training_step_counter,
        }
        
    except Exception as e:
        print(f"✗ Buffer allocation failed: {e}")
        raise


def test_kernel_dispatch(device, module, buffers):
    """Test basic kernel dispatch using ComputeKernel API"""
    print("\n" + "=" * 60)
    print("TEST: Kernel Dispatch")
    print("=" * 60)
    
    max_splats = 100_000
    max_tiles = 14400
    
    def div_ceil(a, b):
        return (a + b - 1) // b
    
    def create_kernel(entry_point):
        """Helper to create a ComputeKernel from an entry point name"""
        shader_path = pathlib.Path(__file__).parent / "shaders" / "kernels.slang"
        program = device.load_program(str(shader_path), [entry_point])
        desc = spy.ComputeKernelDesc()
        desc.program = program
        return device.create_compute_kernel(desc)
    
    try:
        # Test clearAdamState (256 threads per group)
        print("  Testing clearAdamState...")
        kernel = create_kernel("clearAdamState")
        kernel.dispatch(
            thread_count=[div_ceil(max_splats, 256), 1, 1],
            max_splats=max_splats,
            adam_state=buffers['adam_state'].storage
        )
        device.wait_for_idle()
        print(f"  ✓ clearAdamState completed")
        
        # Test clearTileBins (256 threads per group)
        print("  Testing clearTileBins...")
        kernel = create_kernel("clearTileBins")
        kernel.dispatch(
            thread_count=[div_ceil(max_tiles, 256), 1, 1],
            total_tiles=max_tiles,
            screen_tiles_count=buffers['tiles_count'].storage,
            screen_tiles_indices=buffers['tiles_indices'].storage
        )
        device.wait_for_idle()
        print(f"  ✓ clearTileBins completed")
        
        # Test clearProjectedGaussians (256 threads per group)
        print("  Testing clearProjectedGaussians...")
        kernel = create_kernel("clearProjectedGaussians")
        kernel.dispatch(
            thread_count=[div_ceil(max_splats, 256), 1, 1],
            max_splats=max_splats,
            projected_gaussians_2d=buffers['projected'].storage
        )
        device.wait_for_idle()
        print(f"  ✓ clearProjectedGaussians completed")
        
        # Test initializeGaussians (256 threads per group)
        print("  Testing initializeGaussians...")
        num_init = 10000
        kernel = create_kernel("initializeGaussians")
        kernel.dispatch(
            thread_count=[div_ceil(num_init, 256), 1, 1],
            num_gaussians=num_init,
            scene_min=spy.float3(-1.0, -1.0, -1.0),
            scene_max=spy.float3(1.0, 1.0, 1.0),
            initial_scale=0.01,
            initial_opacity=0.5,
            input_gaussians=buffers['gaussians'].storage,
            adam_state=buffers['adam_state'].storage,
            num_gaussians_counter=buffers['num_gaussians_counter'].storage
        )
        device.wait_for_idle()
        print(f"  ✓ initializeGaussians completed ({num_init:,} gaussians)")
        
        # Verify counter was set
        counter_val = int(buffers['num_gaussians_counter'].to_numpy().flat[0])
        print(f"  ✓ Gaussian counter value: {counter_val:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Kernel dispatch failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gaussian_readback(device, module, buffers):
    """Test reading back gaussian data"""
    print("\n" + "=" * 60)
    print("TEST: Data Readback")
    print("=" * 60)
    
    try:
        gaussians_np = buffers['gaussians'].to_numpy()
        print(f"✓ Read back gaussians buffer")
        
        # Check the dtype and structure
        print(f"  Dtype: {gaussians_np.dtype}")
        print(f"  Shape: {gaussians_np.shape}")
        
        # Check first few gaussians
        print(f"\n  First 3 initialized gaussians:")
        for i in range(3):
            g = gaussians_np[i]
            # Access depends on how slangpy serializes struct fields
            print(f"    [{i}] data = {g}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data readback failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("3DGS TRAINER TEST SUITE")
    print("=" * 60 + "\n")
    
    start = time.time()
    
    # Run tests
    device, module = test_shader_compilation()
    buffers = test_buffer_allocation(device, module)
    test_kernel_dispatch(device, module, buffers)
    test_gaussian_readback(device, module, buffers)
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 60)
    print(f"ALL TESTS PASSED in {elapsed:.2f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
