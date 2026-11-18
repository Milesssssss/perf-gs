"""Calibrate synthetic, registered training images without reading a held-out target."""
import struct

import numpy as np
import pytest

from trainer.data.lens_calibration import fit_lens_strength
from trainer.frontend.cli import parse_args


class HeldOutImage:
    def __getattr__(self, key):
        raise AssertionError(f'Calibrator accessed held-out image property {key}')

    def __getitem__(self, key):
        raise AssertionError('Calibrator sampled a held-out image')


def plane_sequence(directory, strength, *, moving=True, mixed_sensors=False):
    """Project a textured plane through translated pinholes under varying exposures."""
    sparse = directory / 'sparse' / '0'
    sparse.mkdir(parents=True)
    w, h = 240, 160
    fx, fy, cx, cy = 145., 127., 109.75, 86.25
    # The supplied RGB is half resolution. The feature file uses native sensor pixels.
    with (sparse / 'cameras.bin').open('wb') as file:
        sensors = (12, 13) if mixed_sensors else (12,)
        file.write(struct.pack('<Q', len(sensors)))
        for sensor in sensors:
            file.write(struct.pack('<iIQQ4d', sensor, 1, 2*w, 2*h, 2*fx, 2*fy, 2*cx, 2*cy))
    y, x = np.mgrid[:h, :w] + .5
    gain = (1 + ((x-cx)/fx)**2 + ((y-cy)/fy)**2)**(-2*strength/2.2)
    land_y, land_x = np.mgrid[8:h-8:7, 8:w-8:7].reshape(2, -1) + .5
    shifts = [(sx, sy) for sy in (-13, 0, 16) for sx in (-39, -13, 17, 37)]
    train_images = {}
    targets = [None] * len(shifts)
    with (sparse / 'images.bin').open('wb') as file:
        file.write(struct.pack('<Q', len(shifts)))
        for view, (sx, sy) in enumerate(shifts):
            if not moving:
                sx = sy = 0
            name = f'frame{view:03d}.png'
            sensor = sensors[view % len(sensors)]
            file.write(struct.pack('<i7di', 400+view, 1, 0, 0, 0, -sx/fx, -sy/fy, 0, sensor))
            file.write(name.encode() + b'\x00')
            u, v = land_x-sx, land_y-sy
            ids = np.flatnonzero((u >= 4) & (v >= 4) & (u < w-4) & (v < h-4))
            file.write(struct.pack('<Q', len(ids)))
            for i in ids:
                file.write(struct.pack('<ddq', 2*u[i], 2*v[i], 1000+i))
            target_index = len(shifts) - 1 - view
            if view in (0, 8):
                targets[target_index] = HeldOutImage()
                continue
            train_images[name] = target_index
            reflectance = .46 + .07*np.cos((x+sx)/19) + .085*np.sin((y+sy)/29)
            exposure = .72 + .09*((view*5) % 7)
            targets[target_index] = (reflectance[..., None] * exposure * gain[..., None]
                                     * np.array([.91, 1, 1.08])).astype(np.float32)
    return train_images, targets


@pytest.mark.parametrize('truth', [0., .65])
def test_recovers_shared_falloff_with_unknown_frame_exposures(tmp_path, truth):
    train_images, targets = plane_sequence(tmp_path, truth)
    fitted, support = fit_lens_strength(tmp_path, train_images, targets)
    assert support >= 60
    assert fitted == pytest.approx(truth, abs=.02)


def test_repeated_view_has_no_radial_evidence(tmp_path):
    train_images, targets = plane_sequence(tmp_path, .65, moving=False)
    assert fit_lens_strength(tmp_path, train_images, targets) == (0., 0)


def test_independent_sensors_do_not_share_one_calibration(tmp_path):
    train_images, targets = plane_sequence(tmp_path, .65, mixed_sensors=True)
    assert fit_lens_strength(tmp_path, train_images, targets) == (0., 0)


def test_missing_sparse_model_uses_identity(tmp_path):
    assert fit_lens_strength(tmp_path, {}, []) == (0., 0)


def test_cli_selects_calibrated_or_manual_strength():
    assert not parse_args([]).auto_lens
    assert parse_args(['--auto-lens']).auto_lens
    assert parse_args(['--lens-strength', '.65']).lens_strength == .65
    with pytest.raises(SystemExit):
        parse_args(['--auto-lens', '--lens-strength', '.65'])
