"""Continuum, symmetry and time-integration checks of the reduced model."""

import numpy as np
import pytest
from nsinduct.shell import (
    dissipation_rates,
    divergence_poloidal_fd,
    initial_omega,
    make_grid,
    negative_curl_curl_toroidal,
    poloidal_field,
    rhs,
    rk4_step,
    run_case,
    volume_integral,
    volume_rms,
    winding_operator,
)
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn


def setup(n=16):
    g = make_grid(dict(ntheta=n, nr=3 * n // 2))
    p = poloidal_field(g, 0.07)
    return g, p["br"], p["btheta"], initial_omega(g, {})


def test_no_shear_and_linear_winding():
    g, br, bt, o = setup()
    z = np.zeros_like(o)
    assert np.max(abs(winding_operator(g, br, bt, np.ones_like(o)))) == 0
    source = winding_operator(g, br, bt, o)
    oo, q = o, z
    for _ in range(20):
        oo, q = rk4_step(g, oo, q, br, bt, 0.01, 0, 0, False)
    np.testing.assert_allclose(oo, o, atol=0, rtol=0)
    np.testing.assert_allclose(q, 0.2 * source, atol=1e-15)
    np.testing.assert_allclose(
        winding_operator(g, 2 * br, 2 * bt, 3 * o), 6 * source, atol=1e-15
    )


def test_analytic_boundary_compatible_winding_refinement():
    errors = []
    for n in [16, 32, 64]:
        g, br, bt, _ = setup(n)
        x = (g.radius2d - g.r_inner) / g.shell_length
        o = np.cos(2 * np.pi * x) * (3 * np.cos(g.theta2d) ** 2 - 1) / 2
        exact = (
            br
            * (-2 * np.pi / g.shell_length * np.sin(2 * np.pi * x))
            * (3 * np.cos(g.theta2d) ** 2 - 1)
            / 2
        )
        exact += (
            bt
            / g.radius2d
            * np.cos(2 * np.pi * x)
            * (-3 * np.sin(g.theta2d) * np.cos(g.theta2d))
        )
        errors.append(volume_rms(g, winding_operator(g, br, bt, o) - exact))
    assert errors[0] / errors[1] > 3.5 and errors[1] / errors[2] > 3.5


def test_divergence_independent_difference_refines():
    errors = []
    for n in [16, 32, 64]:
        g, br, bt, _ = setup(n)
        errors.append(volume_rms(g, divergence_poloidal_fd(g, br, bt)))
    assert errors[0] > errors[1] > errors[2]


def test_exchange_torque_and_dissipation():
    g, br, bt, o = setup()
    q = np.sin(g.theta2d) ** 2 * np.sin(
        np.pi * (g.radius2d - g.r_inner) / g.shell_length
    )
    for eta, nu in [(0, 0), (0.001, 0), (0, 0.001), (0.001, 0.001)]:
        ot, qt = rhs(g, o, q, br, bt, eta, nu, True)
        rate = np.sum(g.mass_s2 * (o * ot + q * qt))
        d = dissipation_rates(g, o, q, eta, nu)
        assert abs(rate + d["ohmic"] + d["viscous"]) < 1e-13
        assert abs(np.sum(g.mass_s2 * ot)) < 1e-13
        assert d["ohmic"] >= 0 and d["viscous"] >= 0
    ot, _ = rhs(g, o, q, br, bt, 0, 0, True)
    assert np.max(abs(ot)) > 0


def test_ohmic_operator_has_no_checkerboard_nullspace():
    g, _, _, _ = setup()
    f = (-1.0) ** np.indices(g.s.shape).sum(axis=0) * g.s
    op, *_ = negative_curl_curl_toroidal(g, f)
    assert -volume_integral(g, f * op) > 100 * volume_integral(g, f * f)


def radial_mode(g):
    ri, ro = g.r_inner, g.r_outer

    def determinant(k):
        return spherical_jn(1, k * ri) * spherical_yn(1, k * ro) - spherical_yn(
            1, k * ri
        ) * spherical_jn(1, k * ro)

    ks = np.linspace(0.1, 20, 1000)
    pairs = [
        (a, b) for a, b in zip(ks[:-1], ks[1:]) if determinant(a) * determinant(b) < 0
    ]
    k = brentq(determinant, *pairs[0])
    r = g.radius2d
    f = spherical_jn(1, k * r) - spherical_jn(1, k * ri) / spherical_yn(
        1, k * ri
    ) * spherical_yn(1, k * r)
    return k, f / r


@pytest.mark.slow
def test_resistive_continuum_eigenmode():
    errors = []
    for n in [16, 32, 64]:
        g, br, bt, _ = setup(n)
        k, q0 = radial_mode(g)
        q = q0.copy()
        o = np.zeros_like(q)
        eta = 0.01
        T = 0.2
        dt = 0.0002
        for _ in range(round(T / dt)):
            o, q = rk4_step(
                g, o, q, np.zeros_like(br), np.zeros_like(bt), dt, eta, 0, False
            )
        exact = q0 * np.exp(-eta * k * k * T)
        errors.append(volume_rms(g, g.s * (q - exact)) / volume_rms(g, g.s * exact))
    assert errors[-1] < 2e-4
    assert errors[0] / errors[1] > 3 and errors[1] / errors[2] > 3


def test_rk4_temporal_order_against_matrix_exponential():
    g, br, bt, o = setup(8)
    shape = o.shape
    N = o.size

    def action(v):
        a, b = rhs(g, v[:N].reshape(shape), v[N:].reshape(shape), br, bt, 0, 0, True)
        return np.r_[a.ravel(), b.ravel()]

    # Small dense matrix assembled independently of RK4, exact semidiscrete evolution.
    from scipy.linalg import expm

    eye = np.eye(2 * N)
    A = np.column_stack([action(v) for v in eye])
    y0 = np.r_[o.ravel(), np.zeros(N)]
    T = 2.0
    ref = expm(T * A) @ y0
    errors = []
    for dt in [0.2, 0.1, 0.05]:
        oo = o.copy()
        q = np.zeros_like(o)
        for _ in range(round(T / dt)):
            oo, q = rk4_step(g, oo, q, br, bt, dt, 0, 0, True)
        errors.append(np.linalg.norm(np.r_[oo.ravel(), q.ravel()] - ref))
    assert errors[0] / errors[1] > 14 and errors[1] / errors[2] > 14


def test_output_times_and_invalid_parameters():
    r = run_case(
        dict(
            ntheta=8,
            nr=12,
            stop_time=0.13,
            timestep=0.03,
            output_dt=0.05,
            snapshot_count=3,
        )
    )
    assert [s["time"] for s in r["snapshots"]] == [0, 0.065, 0.13]
    assert [row["time"] for row in r["rows"]] == [0, 0.05, 0.1, 0.13]
    with pytest.raises(ValueError):
        run_case(dict(timestep=0))
    with pytest.raises(ValueError):
        run_case(dict(timestep=np.nan))


def test_viscosity_stressfree_smoothing_and_prescribed_power():
    g, br, bt, o = setup()
    from nsinduct.shell import negative_weighted_laplacian_omega

    op, *_ = negative_weighted_laplacian_omega(g, np.ones_like(o))
    assert np.max(abs(op)) == 0
    checker = (-1.0) ** np.indices(g.s.shape).sum(axis=0)
    op, *_ = negative_weighted_laplacian_omega(g, checker)
    assert -np.sum(g.mass_s2 * checker * op) > 100 * np.sum(g.mass_s2 * checker**2)
    res = run_case(
        dict(ntheta=8, nr=12, stop_time=1, timestep=0.01, eta=0, lorentz_feedback=False)
    )
    last = res["rows"][-1]
    assert last["prescribed_flow_power"] > 0
    assert abs(last["cumulative_energy_residual"]) < 1e-12


def test_continuum_maxwell_force_on_smooth_interior():
    errors = []
    for n in [24, 48, 96]:
        g, br, bt, o = setup(n)
        theta = g.theta2d
        r = g.radius2d
        k = np.pi / g.shell_length
        x = k * (r - g.r_inner)
        q = np.sin(theta) ** 2 * np.sin(x)
        qr = np.sin(theta) ** 2 * k * np.cos(x)
        qt = 2 * np.sin(theta) * np.cos(theta) * np.sin(x)
        exact = br * (qr + 2 * q / r) + bt / r * (qt + 2 / np.tan(theta) * q)
        actual, _ = rhs(g, o * 0, q, br, bt, 0, 0, True)
        mask = (
            (theta > 0.3)
            & (theta < np.pi - 0.3)
            & (r > g.r_inner + 0.06)
            & (r < g.r_outer - 0.06)
        )
        errors.append(np.sqrt(np.mean((actual[mask] - exact[mask]) ** 2)))
    assert errors[0] / errors[1] > 3 and errors[1] / errors[2] > 3


def test_polar_maxwell_force_and_global_winding_refine():
    force_errors = []
    winding_errors = []
    for n in [24, 48, 96]:
        g, br, bt, o = setup(n)
        r = g.radius2d
        t = g.theta2d
        ot, _ = rhs(g, o * 0, np.ones_like(o), br, bt, 0, 0, True)
        exact = 2 * br / r + 2 * bt / (r * np.tan(t))
        force_errors.append(np.max(abs(ot - exact)))
        k = 2 * np.pi / g.shell_length
        o = np.cos(k * (r - g.r_inner)) * (3 * np.cos(t) ** 2 - 1) / 2
        source = br * (-k * np.sin(k * (r - g.r_inner))) * (3 * np.cos(t) ** 2 - 1) / 2
        source += bt / r * np.cos(k * (r - g.r_inner)) * (-3 * np.sin(t) * np.cos(t))
        winding_errors.append(np.max(abs(winding_operator(g, br, bt, o) - source)))
    assert (
        min(force_errors[0] / force_errors[1], force_errors[1] / force_errors[2]) > 3.7
    )
    assert (
        min(
            winding_errors[0] / winding_errors[1], winding_errors[1] / winding_errors[2]
        )
        > 3.5
    )
    g, br, bt, o = setup()
    with pytest.raises(ValueError):
        winding_operator(g, br, bt * 2, o)


def test_nonconstant_q_global_continuum_force():
    errors = []
    for n in [16, 32, 64]:
        g, br, bt, o = setup(n)
        r = g.radius2d
        t = g.theta2d
        k = np.pi / g.shell_length
        x = k * (r - g.r_inner)
        p2 = (3 * np.cos(t) ** 2 - 1) / 2
        q = np.sin(x) * p2
        qr = k * np.cos(x) * p2
        qt = -3 * np.sin(x) * np.sin(t) * np.cos(t)
        exact = br * (qr + 2 * q / r) + bt / r * (qt + 2 * q / np.tan(t))
        actual, _ = rhs(g, o * 0, q, br, bt, 0, 0, True)
        errors.append(np.max(abs(actual - exact)))
    assert min(errors[0] / errors[1], errors[1] / errors[2]) > 3
