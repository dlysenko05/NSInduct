"""Differential rotation constant on poloidal flux surfaces must not wind."""

import numpy as np
from nsinduct.shell import make_grid, poloidal_field, volume_rms, winding_operator


def test_flux_aligned_differential_rotation_has_zero_continuum_source():
    errors = []
    for n in [16, 32, 64]:
        grid = make_grid(dict(ntheta=n, nr=3 * n // 2))
        field = poloidal_field(grid, 0.07)
        omega = field["psi"] / np.max(field["psi"])
        source = winding_operator(grid, field["br"], field["btheta"], omega)
        assert np.ptp(omega) > 0.9
        errors.append(volume_rms(grid, source))
    assert errors[0] / errors[1] > 3.8 and errors[1] / errors[2] > 3.8
    assert errors[-1] < 2.5e-4
