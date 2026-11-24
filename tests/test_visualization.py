"""Independent physical checks of axisymmetric Cartesian reconstruction."""

import numpy as np
import pytest
from nsinduct.shell import initial_omega, make_grid, poloidal_field, snapshot_fields
from nsinduct.visualization import (
    CartesianField,
    fixed_color_limits,
    integrate_streamline,
    spherical_to_cartesian,
)


def positions(r, theta, phi):
    return np.stack(
        np.broadcast_arrays(
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ),
        axis=-1,
    )


@pytest.fixture
def source():
    g = make_grid(dict(ntheta=24, nr=36))
    p = poloidal_field(g, 0.07)
    q = np.sin(np.pi * (g.radius2d - g.r_inner) / g.shell_length) * np.cos(g.theta2d)
    omega = initial_omega(g, {})
    fields = snapshot_fields(g, omega, q, p["psi"], p["br"], p["btheta"])
    return g, {"time": 1.2345, "fields": fields}


def test_orthonormal_basis_signs_and_norm():
    theta = np.array([np.pi / 2, np.pi / 2, 0, np.pi / 2])
    phi = np.array([0, np.pi / 2, 0, 0])
    np.testing.assert_allclose(
        spherical_to_cartesian(np.zeros(4), np.zeros(4), np.ones(4), theta, phi),
        [[0, 1, 0], [-1, 0, 0], [0, 1, 0], [0, 1, 0]],
        atol=2e-15,
    )
    np.testing.assert_allclose(
        spherical_to_cartesian(np.zeros(4), np.ones(4), np.zeros(4), theta, phi),
        [[0, 0, -1], [0, 0, -1], [1, 0, 0], [0, 0, -1]],
        atol=2e-15,
    )
    br = np.array([0.1, -0.4, 0.3, 1.0])
    bt = np.array([1.0, 2.0, -0.3, 0.1])
    bp = np.array([-0.2, 0.3, 0.4, -0.1])
    cart = spherical_to_cartesian(br, bt, bp, theta, phi)
    assert cart.shape == (4, 3) and np.all(np.isfinite(cart))
    np.testing.assert_allclose(
        np.sum(cart**2, axis=-1), br**2 + bt**2 + bp**2, rtol=2e-15
    )


def test_native_nodes_regular_scalars_and_vector_components(source):
    g, snap = source
    field = CartesianField(snap)
    ii = np.array([0, 4, 10, 18, 23])
    jj = np.array([0, 8, 19, 27, 35])
    r = g.radius2d[ii, jj]
    theta = g.theta2d[ii, jj]
    phi = np.array([0, 0.3, np.pi / 2, 2.0, 5.0])
    sampled = field.sample(positions(r, theta, phi))
    assert field.time == snap["time"]
    assert sampled["B"].shape == (5, 3) and sampled["u"].shape == (5, 3)
    assert np.all(sampled["mask"])
    for name, key in [("q", "q"), ("omega", "omega"), ("Bphi", "B_phi")]:
        np.testing.assert_allclose(
            sampled[name], snap["fields"][key][ii, jj], rtol=3e-13, atol=3e-14
        )
    ephi = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros(5)))
    np.testing.assert_allclose(
        np.sum(sampled["B"] * ephi, axis=1), g.s[ii, jj] * sampled["q"], atol=3e-14
    )
    np.testing.assert_allclose(
        sampled["u"], g.s[ii, jj, None] * sampled["omega"][:, None] * ephi, atol=3e-14
    )
    expected = spherical_to_cartesian(
        snap["fields"]["B_r"][ii, jj],
        snap["fields"]["B_theta"][ii, jj],
        snap["fields"]["B_phi"][ii, jj],
        theta,
        phi,
    )
    np.testing.assert_allclose(sampled["B"], expected, rtol=3e-13, atol=3e-14)


def test_azimuthal_covariance_and_linear_scalar_interpolation(source):
    g, snap = source
    field = CartesianField(snap)
    i, j = 7, 14
    r = (g.radius[j] + g.radius[j + 1]) / 2
    theta = (g.theta[i] + g.theta[i + 1]) / 2
    phi = np.arange(4) * np.pi / 2
    values = field.sample(positions(r, theta, phi))
    for name in ["q", "omega", "Bphi"]:
        np.testing.assert_allclose(
            values[name], np.repeat(values[name][0], 4), atol=2e-14
        )
    for name in ["q", "omega"]:
        expected = np.mean(snap["fields"][name][i : i + 2, j : j + 2])
        np.testing.assert_allclose(values[name], expected, rtol=2e-13, atol=2e-14)
    for name in ["B", "u"]:
        norms = np.linalg.norm(values[name], axis=1)
        np.testing.assert_allclose(
            norms, np.repeat(norms[0], 4), rtol=2e-13, atol=2e-14
        )
        np.testing.assert_allclose(
            values[name][1],
            [-values[name][0, 1], values[name][0, 0], values[name][0, 2]],
            atol=2e-14,
        )


def test_axis_walls_and_outside_mask(source):
    g, snap = source
    field = CartesianField(snap)
    p = positions(
        np.array([0.6, 0.6, 0.3, 1.0, 0.2, 1.1, 0]),
        np.array([0, np.pi, 0.7, 1.2, 0.7, 0.7, 0]),
        0.0,
    )
    sampled = field.sample(p)
    assert np.array_equal(
        sampled["mask"], [True, True, True, True, False, False, False]
    )
    assert np.all(np.isfinite(sampled["B"][:4]))
    assert np.all(np.isfinite(sampled["u"][:4]))
    np.testing.assert_allclose(sampled["Bphi"][:4], 0, atol=3e-14)
    np.testing.assert_allclose(sampled["u"][:2], 0, atol=3e-14)
    np.testing.assert_allclose(sampled["B"][2:4], 0, atol=3e-14)
    assert np.all(np.isnan(sampled["B"][4:]))
    assert np.all(np.isnan(sampled["q"][4:]))


def test_global_signed_scales_use_all_frames(source):
    _, first = source
    second = {"time": 2.5, "fields": dict(first["fields"])}
    second["fields"]["B_phi"] = -2 * first["fields"]["B_phi"]
    second["fields"]["omega"] = 3 * first["fields"]["omega"]
    limits = fixed_color_limits([first, second])
    for name, key in [("Bphi", "B_phi"), ("omega", "omega")]:
        vmax = max(np.max(abs(s["fields"][key])) for s in [first, second])
        np.testing.assert_allclose(limits[name], [-vmax, vmax], rtol=2e-15)


def test_full_streamline_tangents_follow_cartesian_field(source):
    _, snap = source
    field = CartesianField(snap)
    seed = positions(0.65, 0.9, 0.3)
    line = integrate_streamline(
        field, seed, max_length=1.2, step=0.004, direction="both"
    )
    assert line.ndim == 2 and line.shape[1] == 3 and len(line) > 20
    assert np.all(np.isfinite(line))
    radii = np.linalg.norm(line, axis=1)
    assert np.all((radii >= 0.3 - 1e-10) & (radii <= 1.0 + 1e-10))
    delta = np.diff(line, axis=0)
    mid = (line[:-1] + line[1:]) / 2
    sample = field.sample(mid)
    b = sample["B"]
    length = np.linalg.norm(delta, axis=1)
    bnorm = np.linalg.norm(b, axis=1)
    usable = sample["mask"] & (length > 1e-10) & (bnorm > 1e-10)
    assert np.count_nonzero(usable) > 20
    cosine = np.sum(delta[usable] * b[usable], axis=1) / (
        length[usable] * bnorm[usable]
    )
    assert np.min(cosine) > 0.999
    azimuth = np.unwrap(np.arctan2(line[:, 1], line[:, 0]))
    assert np.ptp(azimuth) > 0.1
    sin2 = 1 - (line[:, 2] / radii) ** 2
    flux = ((radii - 0.3) * (1.0 - radii)) ** 2 * sin2
    assert np.max(abs(flux - flux[0])) / abs(flux[0]) < 1e-5


def test_saved_production_times_ranges_and_energy_are_preserved(tmp_path):
    import h5py
    from nsinduct.shell import run_case, volume_integral

    config = dict(
        ntheta=8,
        nr=12,
        stop_time=0.13,
        timestep=0.03,
        output_dt=0.065,
        snapshot_count=3,
    )
    result = run_case(config, tmp_path)
    grid = make_grid(config)
    with h5py.File(tmp_path / "snapshots.h5") as store:
        for group, row in zip((store[name] for name in sorted(store)), result["rows"]):
            snap = {
                "time": float(group.attrs["time"]),
                "fields": {name: group[name][...] for name in group},
            }
            field = CartesianField(snap)
            assert field.time == row["time"]
            assert field.time in [0.0, 0.065, 0.13]
            xyz = positions(grid.radius2d, grid.theta2d, 0.0).reshape(-1, 3)
            samples = field.sample(xyz)
            bphi = samples["Bphi"].reshape(grid.s.shape)
            omega = samples["omega"].reshape(grid.s.shape)
            np.testing.assert_allclose(
                np.max(abs(bphi)), row["max_abs_B_phi"], atol=3e-14
            )
            np.testing.assert_allclose(
                [omega.min(), omega.max()],
                [row["min_omega"], row["max_omega"]],
                atol=3e-14,
            )
            np.testing.assert_allclose(
                0.5 * volume_integral(grid, bphi**2),
                row["E_tor"],
                rtol=1e-12,
                atol=1e-16,
            )
            velocity2 = np.sum(samples["u"] ** 2, axis=1).reshape(grid.s.shape)
            np.testing.assert_allclose(
                0.5 * volume_integral(grid, velocity2), row["E_kin"], rtol=1e-12
            )


def test_equatorial_parity_of_symmetric_source(source):
    _, snap = source
    field = CartesianField(snap)
    r = np.array([0.4, 0.6, 0.85])
    theta = np.array([0.4, 0.9, 1.2])
    phi = 0.7
    north = field.sample(positions(r, theta, phi))
    south = field.sample(positions(r, np.pi - theta, phi))
    np.testing.assert_allclose(south["omega"], north["omega"], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(south["q"], -north["q"], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(south["Bphi"], -north["Bphi"], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(south["u"], north["u"], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(
        south["B"][:, :2], -north["B"][:, :2], rtol=2e-13, atol=2e-14
    )
    np.testing.assert_allclose(
        south["B"][:, 2], north["B"][:, 2], rtol=2e-13, atol=2e-14
    )


def test_bounded_polar_continuation_and_unsupported_geometry(source):
    _, snap = source
    field = CartesianField(snap)
    radius, theta = np.meshgrid(np.linspace(0.3, 1.0, 53), np.linspace(0, np.pi, 71))
    sampled = field.sample(positions(radius, theta, 0.4).reshape(-1, 3))
    for key in ["q", "omega"]:
        lo = (
            min(float(np.min(snap["fields"][key])), 0.0)
            if key == "q"
            else float(np.min(snap["fields"][key]))
        )
        hi = (
            max(float(np.max(snap["fields"][key])), 0.0)
            if key == "q"
            else float(np.max(snap["fields"][key]))
        )
        assert np.min(sampled[key]) >= lo - 2e-14
        assert np.max(sampled[key]) <= hi + 2e-14
    mismatched = {"time": snap["time"], "fields": dict(snap["fields"])}
    mismatched["fields"]["B_theta"] = snap["fields"]["B_theta"] + 0.01
    with pytest.raises(ValueError):
        CartesianField(mismatched)
