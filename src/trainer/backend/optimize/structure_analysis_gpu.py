"""GPU port of the multi-scale structure-tensor field (structure_analysis.py).

The CPU reference (:func:`trainer.structure_analysis.compute_structure_tensor_field`)
builds the per-view aggregated structure tensor S-bar with scipy — clear and
testable, but it does ~20 Gaussian blurs per image in float64 on the CPU, so a
194-view scene takes minutes. This module runs the identical pipeline as a
chain of separable-Gaussian / Sobel / DoG compute passes
(``shaders/structure/structure_tensor.slang``), building each field in a few ms.

The CPU version stays the reference oracle; :class:`StructureTensorGPU` is a
drop-in that returns the same ``(H, W, 3)`` float32 ``(Sxx, Sxy, Syy)`` array.
See ``tests/test_structure_analysis_gpu.py`` for the parity check.

Usage (reuse one instance across a scene's views — kernels + textures are
allocated once and recycled per image)::

    st = StructureTensorGPU(device, width, height)
    for img in images:            # img: (H, W, 3|4) sRGB float32 in [0,1]
        sbar = st.compute(img)    # (H, W, 3) float32
"""

from __future__ import annotations

import math

import numpy as np
import slangpy as spy

from trainer.backend.compute.device import SHADERS_DIR
from trainer.backend.compute.gpu import GpuRuntime
from trainer.backend.optimize.structure_analysis import (
    GAMMA,
    MIN_ENERGY,
    NUM_LEVELS,
    OMEGA_CALIBRATION,
    RHO_FACTOR,
    SIGMA_BASE,
)

_SHADER = "structure/structure_tensor.slang"
# scipy.ndimage.gaussian_filter default truncate; radius = int(truncate*sigma + 0.5).
_TRUNCATE = 4.0

# All entry points in the shader, loaded into self.k_<name>.
_ENTRIES = (
    "blurH", "blurV", "sobelTensor", "clearMaxTrace", "reduceMaxTrace",
    "normalizeBase", "dogBand", "clearAccum", "accumulateLevel", "finalize",
)


def _blur_radius(sigma: float) -> int:
    """scipy's Gaussian kernel radius for a given sigma (truncate=4.0)."""
    return int(_TRUNCATE * float(sigma) + 0.5)


class StructureTensorGPU:
    """GPU builder for the aggregated structure-tensor field S-bar.

    Fixed to one (width, height); allocate one per resolution and reuse it for
    every view of the scene. Not thread-safe (shares the scratch textures /
    command submission across calls) — drive it from a single thread.
    """

    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
        num_levels: int = NUM_LEVELS,
        sigma_base: float = SIGMA_BASE,
        rho_factor: float = RHO_FACTOR,
        gamma: float = GAMMA,
    ) -> None:
        self.device = device
        self.runtime = GpuRuntime(device)
        self.width = int(width)
        self.height = int(height)
        self.num_levels = int(num_levels)
        self.sigma_base = float(sigma_base)
        self.rho_factor = float(rho_factor)
        self.gamma = float(gamma)

        self._dim = spy.uint2(self.width, self.height)
        self._tc = [self.width, self.height, 1]

        # Compile kernels.
        for name in _ENTRIES:
            program = device.load_program(str(SHADERS_DIR / _SHADER), [name])
            setattr(self, f"k_{name}", device.create_compute_kernel(program))

        # Working textures (rgba32_float, W x H). Reused across every compute().
        def _tex(label: str) -> spy.Texture:
            return device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spy.Format.rgba32_float,
                width=self.width, height=self.height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label=f"st_{label}",
            )

        self.img_tex = _tex("img")        # input sRGB (rgb in .xyz)
        self.scratchA = _tex("scratchA")  # separable-blur intermediate
        self.scratchB = _tex("scratchB")  # base-tensor smoothed image
        self.base_tex = _tex("base")      # normalized base orientation tensor
        self.int_tex = _tex("int")        # base tensor re-integrated per level
        self.band_tex = _tex("band")      # DoG band response (.x)
        self.smooth0 = _tex("smooth0")    # DoG cascade ping-pong
        self.smooth1 = _tex("smooth1")
        self.accum_tex = _tex("accum")    # (num.xyz, den.w)
        self.out_tex = _tex("out")        # final S-bar (Sxx, Sxy, Syy)

        # 1-element max-trace reduction target.
        self.max_trace = spy.NDBuffer(
            device=device, dtype="uint", shape=(1,),
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )

    # --- dispatch helpers -------------------------------------------------
    def _d(self, enc, kernel, **kw) -> None:
        kernel.dispatch(command_encoder=enc, thread_count=self._tc, **kw)

    def _blur(self, enc, src: spy.Texture, dst: spy.Texture, sigma: float) -> None:
        """Separable Gaussian blur src -> dst via scratchA (mode='nearest').

        In-place (src is dst) is safe: blurH writes scratchA, blurV reads it."""
        radius = _blur_radius(sigma)
        self._d(enc, self.k_blurH, src=src, dst=self.scratchA,
                dim=self._dim, sigma=float(sigma), radius=int(radius))
        self._d(enc, self.k_blurV, src=self.scratchA, dst=dst,
                dim=self._dim, sigma=float(sigma), radius=int(radius))

    # --- main entry -------------------------------------------------------
    def compute(self, image_srgb: np.ndarray) -> np.ndarray:
        """Aggregated structure tensor S-bar for one image.

        ``image_srgb`` is (H, W, 3) or (H, W, 4) RAW sRGB in [0, 1] (alpha
        ignored). Returns (H, W, 3) float32 packing (Sxx, Sxy, Syy) — matching
        the CPU reference's convention exactly. The image resolution must equal
        the (width, height) this instance was built for."""
        img = np.asarray(image_srgb, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError(f"expected (H, W, >=3) image, got {img.shape}")
        H, W = img.shape[:2]
        if (W, H) != (self.width, self.height):
            raise ValueError(
                f"image {W}x{H} != instance {self.width}x{self.height}; "
                "build a StructureTensorGPU per resolution")

        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[..., :3] = img[..., :3]
        self.img_tex.copy_from_numpy(np.ascontiguousarray(rgba))

        batch = self.runtime.command_batch("structure_tensor")
        enc = batch.encoder

        # 1) Base orientation tensor at the finest scale, then normalized by the
        #    batch-max trace (SADGS _structure_tensor).
        self._blur(enc, self.img_tex, self.scratchB, self.sigma_base)  # smooth by sigma_base
        self._d(enc, self.k_sobelTensor, smoothed=self.scratchB,
                tensorOut=self.base_tex, dim=self._dim)
        self._blur(enc, self.base_tex, self.base_tex, 1.0)             # integrate by rho=1
        self.k_clearMaxTrace.dispatch(command_encoder=enc, thread_count=[1, 1, 1],
                                      maxTrace=self.max_trace.storage)
        self._d(enc, self.k_reduceMaxTrace, tensorIn=self.base_tex,
                maxTrace=self.max_trace.storage, dim=self._dim)
        self._d(enc, self.k_normalizeBase, tensor=self.base_tex,
                maxTrace=self.max_trace.storage, dim=self._dim)

        # 2) Multi-scale aggregation over the DoG bands.
        self._d(enc, self.k_clearAccum, accum=self.accum_tex, dim=self._dim)
        cur = self.img_tex
        sigma_accum = 0.0
        for level in range(self.num_levels):
            target_sigma = self.sigma_base ** level
            rho_l = self.rho_factor * target_sigma
            sigma_inc = math.sqrt(max(1e-6, target_sigma ** 2 - sigma_accum ** 2))
            band_freq = 1.0 / (self.sigma_base ** level)
            freq_sq = (band_freq * OMEGA_CALIBRATION) ** 2

            nxt = self.smooth0 if (level % 2 == 0) else self.smooth1
            self._blur(enc, cur, nxt, sigma_inc)                        # I_{l+1}
            self._d(enc, self.k_dogBand, cur=cur, nxt=nxt,
                    bandOut=self.band_tex, dim=self._dim)
            if level > 0:
                self._blur(enc, self.band_tex, self.band_tex, target_sigma * 2.0)
            self._blur(enc, self.base_tex, self.int_tex, rho_l)        # re-integrate
            self._d(enc, self.k_accumulateLevel, band=self.band_tex,
                    intTensor=self.int_tex, accum=self.accum_tex,
                    dim=self._dim, gamma=float(self.gamma), freq_sq=float(freq_sq))

            cur = nxt
            sigma_accum = target_sigma

        self._d(enc, self.k_finalize, accum=self.accum_tex, fieldOut=self.out_tex,
                dim=self._dim, min_energy=float(MIN_ENERGY))

        self.runtime.submit(batch)
        self.runtime.wait_for_idle()

        return np.ascontiguousarray(self.out_tex.to_numpy()[..., :3].astype(np.float32))


def compute_structure_tensor_field_gpu(
    device: spy.Device,
    image_srgb: np.ndarray,
    num_levels: int = NUM_LEVELS,
    sigma_base: float = SIGMA_BASE,
    rho_factor: float = RHO_FACTOR,
    gamma: float = GAMMA,
) -> np.ndarray:
    """One-shot GPU structure-tensor field (allocates + tears down per call).

    Convenience wrapper matching :func:`compute_structure_tensor_field`. For a
    whole scene, prefer building one :class:`StructureTensorGPU` and calling
    ``.compute`` per view so the kernels/textures are reused."""
    H, W = image_srgb.shape[:2]
    return StructureTensorGPU(
        device, W, H, num_levels=num_levels, sigma_base=sigma_base,
        rho_factor=rho_factor, gamma=gamma,
    ).compute(image_srgb)
