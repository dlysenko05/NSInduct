"""Backend contracts and physically meaningful toroidal-subspace checks."""

import numpy as np
import pytest
from nsinduct.dedalus_backend import run_case

pytestmark = pytest.mark.dedalus

pytest.importorskip("dedalus.public")


def test_invalid_and_resource_limits():
    with pytest.raises(ValueError):
        run_case(dict(eta=-1))
    with pytest.raises(ValueError):
        run_case(dict(ntheta=128))
    with pytest.raises(RuntimeError, match="budget"):
        run_case(
            dict(
                ntheta=8, nr=12, eta=0, stop_time=0.03, timestep=0.01, max_iterations=1
            )
        )


def test_native_snapshots_exact_times_and_no_shear():
    result = run_case(
        dict(
            ntheta=12,
            nr=24,
            eta=0,
            lorentz_feedback=False,
            initial_omega=lambda t, r: np.ones_like(t),
            stop_time=0.037,
            timestep=0.01,
            output_dt=0.02,
            snapshot_count=3,
        )
    )
    assert [s["time"] for s in result["snapshots"]] == [0, 0.0185, 0.037]
    assert [r["time"] for r in result["rows"]] == [0, 0.02, 0.037]
    assert result["rows"][-1]["max_abs_B_phi"] < 1e-9
    for snap in result["snapshots"]:
        f = snap["fields"]
        assert f["q"].shape == (12, 24)
        assert np.all(np.diff(f["theta"]) > 0) and np.all(np.diff(f["radius"]) > 0)
        np.testing.assert_allclose(f["B_phi"], f["s"] * f["q"], atol=1e-16)
        np.testing.assert_allclose(f["u_phi"], f["s"] * f["omega"], atol=1e-15)
    assert result["metadata"]["peak_memory_bytes"] > 0


def test_feedback_is_same_setup_as_prescribed_control():
    config = dict(
        ntheta=12,
        nr=24,
        eta=2e-5,
        stop_time=0.5,
        timestep=0.01,
        output_dt=0.1,
        snapshot_count=2,
    )
    evolved = run_case(config)
    control = run_case(config | dict(lorentz_feedback=False))
    for key in ["omega", "q", "B_r", "B_theta"]:
        np.testing.assert_array_equal(
            evolved["snapshots"][0]["fields"][key],
            control["snapshots"][0]["fields"][key],
        )
    assert evolved["rows"][-1]["E_kin"] < evolved["rows"][0]["E_kin"]
    np.testing.assert_allclose(
        control["rows"][-1]["E_kin"], control["rows"][0]["E_kin"], rtol=1e-12
    )
    assert control["rows"][-1]["E_tor"] > evolved["rows"][-1]["E_tor"]
    for result in [evolved, control]:
        assert max(r["max_meridional_U"] for r in result["rows"]) < 1e-13
        assert max(r["max_meridional_T"] for r in result["rows"]) < 1e-13
        assert max(r["toroidal_wall_residual"] for r in result["rows"]) < 1e-10
        assert result["summary"]["max_cumulative_energy_residual"] < 1e-6


def test_viscous_damping_and_angular_momentum():
    result = run_case(
        dict(
            ntheta=12,
            nr=24,
            eta=0.0,
            nu=0.001,
            field_strength=0.0,
            stop_time=0.1,
            timestep=0.005,
            output_dt=0.05,
            snapshot_count=2,
        )
    )
    first, last = result["rows"][0], result["rows"][-1]
    assert last["E_kin"] < first["E_kin"]
    assert last["D_viscous"] > 0
    assert abs(last["angular_momentum_total"] - first["angular_momentum_total"]) < 1e-7
    assert last["physical_viscous_wall_residual"] < 1e-5
    assert result["summary"]["max_cumulative_energy_residual"] < 1e-6


def test_resistive_decay_has_no_spurious_growing_tau_mode():
    result = run_case(
        dict(
            ntheta=24,
            nr=36,
            field_strength=0.0,
            eta=2e-5,
            initial_omega=lambda t, r: 0 * t,
            initial_q=lambda t, r: np.sin(np.pi * (r - 0.3) / 0.7) * np.cos(t),
            lorentz_feedback=False,
            stop_time=0.2,
            timestep=0.01,
            output_dt=0.02,
            snapshot_count=2,
        )
    )
    energies = np.array([r["E_tor"] for r in result["rows"]])
    assert np.all(np.diff(energies) <= 1e-11 * energies[0])
    assert energies[-1] < energies[0]
    assert result["summary"]["max_toroidal_wall_residual"] < 1e-10
    assert result["summary"]["max_cumulative_energy_residual"] < 1e-7


def test_independent_viscous_robin_eigenmode():
    from scipy.optimize import brentq
    from scipy.special import spherical_jn as jn
    from scipy.special import spherical_yn as yn

    def derivative(k, r, fn):
        return r * k * fn(1, k * r, derivative=True) - fn(1, k * r)

    def determinant(k):
        return derivative(k, 0.3, jn) * derivative(k, 1, yn) - derivative(
            k, 0.3, yn
        ) * derivative(k, 1, jn)

    k = brentq(determinant, 5.0, 7.0)
    alpha = -derivative(k, 0.3, jn) / derivative(k, 0.3, yn)

    def initial(t, r):
        return (jn(1, k * r) + alpha * yn(1, k * r)) / r

    result = run_case(
        dict(
            ntheta=8,
            nr=24,
            field_strength=0.0,
            eta=0.0,
            nu=0.01,
            initial_omega=initial,
            lorentz_feedback=False,
            stop_time=0.2,
            timestep=0.002,
            output_dt=0.1,
            snapshot_count=2,
        )
    )
    first, last = result["snapshots"]
    expected = first["fields"]["omega"] * np.exp(-0.01 * k * k * 0.2)
    error = np.max(abs(last["fields"]["omega"] - expected)) / np.max(abs(expected))
    assert error < 1e-8
    assert result["rows"][-1]["physical_viscous_wall_residual"] < 1e-9


def test_prescribed_field_is_immutable_and_near_equal_targets_coalesce():
    from nsinduct.visualization import CartesianField

    result = run_case(
        dict(
            ntheta=12,
            nr=24,
            stop_time=0.3,
            timestep=0.01,
            output_dt=0.1,
            snapshot_count=3,
            eta=2e-5,
        )
    )
    assert len(result["snapshots"]) == 3
    assert len(result["rows"]) == 4
    first = result["snapshots"][0]["fields"]
    for snapshot in result["snapshots"]:
        fields = snapshot["fields"]
        np.testing.assert_array_equal(fields["B_r"], first["B_r"])
        np.testing.assert_array_equal(fields["B_theta"], first["B_theta"])
        CartesianField(snapshot)


def test_field_scaled_stop_time_records_each_snapshot_once():
    stop = 0.98 / 0.07
    result = run_case(
        dict(
            ntheta=8,
            nr=12,
            eta=0.0,
            field_strength=0.0,
            initial_omega=lambda t, r: 0 * t,
            stop_time=stop,
            timestep=1.0,
            output_dt=0.1,
            snapshot_count=3,
        )
    )
    assert len(result["snapshots"]) == 3
    assert len(result["rows"]) == 141
    np.testing.assert_allclose(
        [s["time"] for s in result["snapshots"]],
        [0.0, stop / 2, stop],
        rtol=0,
        atol=1e-12,
    )
