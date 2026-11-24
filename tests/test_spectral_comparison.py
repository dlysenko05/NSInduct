"""Independent continuum checks; optional Dedalus dependency, actual PDE evolution."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.dedalus

_spec = importlib.util.spec_from_file_location(
    "independent_spectral_reference",
    Path(__file__).resolve().parents[1] / "scripts" / "verify_spectral.py",
)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)


def test_independent_continuous_normalization_and_derivatives():
    x, w = np.polynomial.legendre.leggauss(100)
    r = 0.3 + (x + 1) * 0.7 / 2
    wr = w * 0.7 / 2
    rr, mu = np.meshgrid(r, x)
    theta = np.arccos(mu)
    f = _reference.reference_fields(theta, rr)
    weights = 2 * np.pi * w[:, None] * wr[None, :] * rr**2
    moment = weights * rr**2 * (1 - mu**2)
    volume = 4 * np.pi / 3 * (1 - 0.3**3)
    assert (
        abs(
            np.sqrt(np.sum(weights * (f["br"] ** 2 + f["btheta"] ** 2)) / volume) - 0.07
        )
        < 2e-14
    )
    assert abs(np.sum(moment * f["omega"])) < 2e-14
    assert abs(np.sum(moment * f["omega"] ** 2) / np.sum(moment) - 1) < 2e-14
    nodes = np.linspace(0.3, 1, 12)
    derivative = _reference.derivative_matrix(nodes)
    np.testing.assert_allclose(
        derivative @ (nodes**4), 4 * nodes**3, rtol=1e-11, atol=1e-12
    )


@pytest.fixture
def spectral_solver():
    pytest.importorskip("dedalus.public")
    from nsinduct.dedalus_backend import run_case

    return run_case


def base_config():
    config = dict(
        ntheta=16,
        nr=24,
        field_strength=0.07,
        eta=0.0,
        nu=0.0,
        lorentz_feedback=False,
        stop_time=0.1,
        timestep=0.001,
        output_dt=0.05,
        snapshot_count=2,
    )
    config.update(_reference.continuous_normalization())
    config["initial_omega"] = lambda theta, r: _reference.reference_fields(theta, r)[
        "omega"
    ]
    return config


def test_spectral_continuum_ideal_winding(spectral_solver):
    result = spectral_solver(base_config())
    last = result["snapshots"][-1]
    f = last["fields"]
    rr, tt = np.meshgrid(f["radius"], f["theta"])
    exact = last["time"] * _reference.reference_fields(tt, rr)["source"]
    assert last["time"] == pytest.approx(0.1, abs=1e-13)
    assert _reference.error_norm(last, exact) < 1e-6
    assert _reference.divergence_residual(last) < 1e-8


def test_spectral_continuum_resistive_eigenmode(spectral_solver):
    config = base_config()
    config.update(field_strength=0.0, poloidal_scale=0.0, eta=0.01, stop_time=0.2)
    config["initial_omega"] = lambda theta, r: np.zeros(
        np.broadcast_shapes(np.shape(theta), np.shape(r))
    )
    config["initial_q"] = lambda theta, r: np.broadcast_to(
        _reference.decay_mode(r)[1], np.broadcast_shapes(np.shape(theta), np.shape(r))
    ).copy()
    result = spectral_solver(config)
    last = result["snapshots"][-1]
    rr, tt = np.meshgrid(last["fields"]["radius"], last["fields"]["theta"])
    k, q0 = _reference.decay_mode(rr)
    assert k == pytest.approx(5.042726912758328, rel=2e-14)
    exact = q0 * np.exp(-0.01 * k * k * last["time"])
    assert _reference.error_norm(last, exact) < 1e-5


def test_spectral_feedback_independent_energy_and_angular_momentum(spectral_solver):
    config = base_config()
    config.update(lorentz_feedback=True, stop_time=0.2, snapshot_count=3)
    result = spectral_solver(config)
    diagnostics = [_reference.independent_integrals(s) for s in result["snapshots"]]
    e0 = diagnostics[0]["E_kin"] + diagnostics[0]["E_tor"]
    assert diagnostics[-1]["E_tor"] > 0
    assert diagnostics[-1]["E_kin"] < diagnostics[0]["E_kin"]
    assert max(abs(d["E_kin"] + d["E_tor"] - e0) for d in diagnostics) / e0 < 1e-5
    assert max(abs(d["J"] - diagnostics[0]["J"]) for d in diagnostics) < 1e-7


def test_spectral_stressfree_rigid_rotation_viscous_null(spectral_solver):
    config = base_config()
    config.update(
        field_strength=0.0,
        poloidal_scale=0.0,
        nu=0.01,
        lorentz_feedback=True,
        stop_time=0.1,
    )
    config["initial_omega"] = lambda theta, r: np.ones(
        np.broadcast_shapes(np.shape(theta), np.shape(r))
    )
    result = spectral_solver(config)
    for snap in result["snapshots"]:
        np.testing.assert_allclose(snap["fields"]["omega"], 1.0, atol=1e-8, rtol=1e-8)
        assert np.max(abs(snap["fields"]["q"])) < 1e-12


def test_actual_spectral_snapshot_cartesian_mapping_and_field_direction(
    spectral_solver,
):
    from nsinduct.visualization import CartesianField, integrate_streamline

    result = spectral_solver(base_config())
    snap = result["snapshots"][-1]
    f = snap["fields"]
    field = CartesianField(snap)
    ii = np.array([0, 3, 7, 12, 15])
    jj = np.array([0, 5, 11, 18, 23])
    phi = np.array([0, 0.4, 1.2, 2.0, 4.0])
    theta = np.asarray(f["theta"])[ii]
    r = np.asarray(f["radius"])[jj]
    er = np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    et = np.column_stack(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
    )
    ep = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
    sampled = field.sample(r[:, None] * er)
    expected = (
        f["B_r"][ii, jj, None] * er
        + f["B_theta"][ii, jj, None] * et
        + f["B_phi"][ii, jj, None] * ep
    )
    np.testing.assert_allclose(sampled["B"], expected, atol=2e-12, rtol=2e-11)
    np.testing.assert_allclose(
        sampled["u"], f["u_phi"][ii, jj, None] * ep, atol=2e-12, rtol=2e-11
    )
    walls = field.sample(
        [[0.3, 0, 0], [1, 0, 0], [0, 0, 0.6], [0, 0, -0.6], [0.2, 0, 0]]
    )
    assert walls["mask"].tolist() == [True, True, True, True, False]
    np.testing.assert_allclose(walls["Bphi"][:4], 0, atol=2e-13)
    np.testing.assert_allclose(walls["u"][2:4], 0, atol=2e-13)
    seed = np.array([0.65 * np.sin(1.0), 0, 0.65 * np.cos(1.0)])
    points = integrate_streamline(
        field, seed, max_length=0.2, step=0.002, direction="forward"
    )
    tangent = points[2:] - points[:-2]
    b = field.sample(points[1:-1])["B"]
    cosine = np.sum(tangent * b, axis=1) / (
        np.linalg.norm(tangent, axis=1) * np.linalg.norm(b, axis=1)
    )
    assert np.min(cosine) > 0.999
