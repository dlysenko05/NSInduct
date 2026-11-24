"""Independent continuum and matched finite-difference checks of the spectral backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn


def continuous_normalization(field_strength=0.07, contrast=0.18, ri=0.3, ro=1.0):
    """Continuum field and rotation constants from independent Gaussian quadrature."""
    x, w = leggauss(128)
    r = ri + (x + 1) * (ro - ri) / 2
    wr = w * (ro - ri) / 2
    h = ((r - ri) * (ro - r)) ** 2
    hp = 2 * (r - ri) * (ro - r) * (ri + ro - 2 * r)
    volume = 4 * np.pi / 3 * (ro**3 - ri**3)
    bp2 = 2 * np.pi * np.sum(wr * (8 * h**2 / (3 * r**2) + 4 * hp**2 / 3)) / volume
    scale = field_strength / np.sqrt(bp2)
    radial = 1 + contrast * np.cos(2 * np.pi * (r - ri) / (ro - ri))
    radial_mean = np.sum(wr * r**4 * radial) / np.sum(wr * r**4)
    mean = -radial_mean / 5
    second = np.sum(wr * r**4 * radial**2) / np.sum(wr * r**4) / 7
    amplitude = np.sqrt(second - mean**2)
    return dict(
        poloidal_scale=float(scale),
        omega_mean=float(mean),
        omega_amplitude=float(amplitude),
    )


def reference_fields(theta, radius, field_strength=0.07, contrast=0.18, ri=0.3, ro=1.0):
    """Analytic confined field, normalized initial rotation and ideal winding source."""
    theta, r = np.broadcast_arrays(theta, radius)
    norm = continuous_normalization(field_strength, contrast, ri, ro)
    h = ((r - ri) * (ro - r)) ** 2
    hp = 2 * (r - ri) * (ro - r) * (ri + ro - 2 * r)
    br = norm["poloidal_scale"] * 2 * h * np.cos(theta) / r**2
    bt = -norm["poloidal_scale"] * hp * np.sin(theta) / r
    k = 2 * np.pi / (ro - ri)
    radial = 1 + contrast * np.cos(k * (r - ri))
    p2 = (3 * np.cos(theta) ** 2 - 1) / 2
    omega = (radial * p2 - norm["omega_mean"]) / norm["omega_amplitude"]
    omega_r = -contrast * k * np.sin(k * (r - ri)) * p2 / norm["omega_amplitude"]
    omega_theta = -3 * radial * np.sin(theta) * np.cos(theta) / norm["omega_amplitude"]
    return dict(
        br=br, btheta=bt, omega=omega, source=br * omega_r + bt * omega_theta / r
    )


def decay_mode(radius, ri=0.3, ro=1.0):
    """First l=1 toroidal Dirichlet radial eigenmode q=f/r and its exact k."""

    def determinant(k):
        return spherical_jn(1, k * ri) * spherical_yn(1, k * ro) - spherical_yn(
            1, k * ri
        ) * spherical_jn(1, k * ro)

    candidates = np.linspace(0.1, 30, 2000)
    a, b = next(
        (a, b)
        for a, b in zip(candidates[:-1], candidates[1:])
        if determinant(a) * determinant(b) < 0
    )
    k = brentq(determinant, a, b, xtol=1e-14)
    f = spherical_jn(1, k * np.asarray(radius)) - spherical_jn(
        1, k * ri
    ) / spherical_yn(1, k * ri) * spherical_yn(1, k * np.asarray(radius))
    return float(k), f / np.asarray(radius)


def tensor_evaluate(theta, radius, values, mu_target, r_target):
    """Interpolate in cos(theta) and radius, independently of Dedalus operators."""
    radial = BarycentricInterpolator(np.asarray(radius), np.asarray(values), axis=1)(
        r_target
    )
    return BarycentricInterpolator(np.cos(theta), radial, axis=0)(mu_target)


def independent_integrals(snapshot, ri=0.3, ro=1.0, order=80):
    """Evaluate energy and angular momentum with independent Gauss-Legendre quadrature."""
    f = snapshot["fields"]
    mu, wm = leggauss(order)
    x, wx = leggauss(order)
    r = ri + (x + 1) * (ro - ri) / 2
    wr = wx * (ro - ri) / 2
    omega = tensor_evaluate(f["theta"], f["radius"], f["omega"], mu, r)
    q = tensor_evaluate(f["theta"], f["radius"], f["q"], mu, r)
    moment = (
        2 * np.pi * wm[:, None] * wr[None, :] * r[None, :] ** 4 * (1 - mu[:, None] ** 2)
    )
    dr = derivative_matrix(f["radius"])
    dm = derivative_matrix(np.cos(f["theta"]))
    qr = tensor_evaluate(f["theta"], f["radius"], np.asarray(f["q"]) @ dr.T, mu, r)
    qm = tensor_evaluate(f["theta"], f["radius"], dm @ np.asarray(f["q"]), mu, r)
    omega_r = tensor_evaluate(
        f["theta"], f["radius"], np.asarray(f["omega"]) @ dr.T, mu, r
    )
    omega_m = tensor_evaluate(
        f["theta"], f["radius"], dm @ np.asarray(f["omega"]), mu, r
    )
    jr = 2 * mu[:, None] * q - (1 - mu[:, None] ** 2) * qm
    jtheta2 = (1 - mu[:, None] ** 2) * (2 * q + r[None, :] * qr) ** 2
    volume_weights = 2 * np.pi * wm[:, None] * wr[None, :] * r[None, :] ** 2
    ohmic = float(np.sum(volume_weights * (jr**2 + jtheta2)))
    viscous = float(
        np.sum(
            moment
            * (omega_r**2 + (1 - mu[:, None] ** 2) / r[None, :] ** 2 * omega_m**2)
        )
    )
    return dict(
        E_kin=float(np.sum(moment * omega**2) / 2),
        E_tor=float(np.sum(moment * q**2) / 2),
        J=float(np.sum(moment * omega)),
        shear=float(np.sqrt(np.sum(moment * omega**2) / np.sum(moment))),
        ohmic_loss_per_eta=ohmic,
        viscous_loss_per_nu=viscous,
    )


def derivative_matrix(nodes):
    x = np.asarray(nodes)
    weights = BarycentricInterpolator(x).wi
    difference = x[:, None] - x[None, :]
    np.fill_diagonal(difference, 1)
    matrix = weights[None, :] / weights[:, None] / difference
    np.fill_diagonal(matrix, 0)
    np.fill_diagonal(matrix, -matrix.sum(axis=1))
    return matrix


def divergence_residual(snapshot):
    """Differentiate saved physical Bp components; this is not construction-zero."""
    f = snapshot["fields"]
    r = np.asarray(f["radius"])
    mu = np.cos(f["theta"])
    dr = derivative_matrix(r)
    dm = derivative_matrix(mu)
    div = (np.asarray(f["B_r"]) * r[None, :] ** 2) @ dr.T / r[None, :] ** 2
    div -= dm @ (np.sqrt(1 - mu[:, None] ** 2) * f["B_theta"]) / r[None, :]
    return float(np.max(abs(div)))


def error_norm(snapshot, exact_q):
    f = snapshot["fields"]
    diff = dict(snapshot)
    diff["fields"] = dict(
        f, q=np.asarray(f["q"]) - exact_q, omega=np.zeros_like(exact_q)
    )
    ref = dict(snapshot)
    ref["fields"] = dict(f, q=exact_q, omega=np.zeros_like(exact_q))
    numerator = independent_integrals(diff)["E_tor"]
    denominator = independent_integrals(ref)["E_tor"]
    return float(np.sqrt(numerator / denominator))


def analytic_experiments(run_spectral, quick=False):
    records = []
    for n in [12, 16] if quick else [12, 16, 24]:
        config = dict(
            ntheta=n,
            nr=3 * n // 2,
            field_strength=0.07,
            eta=0.0,
            nu=0.0,
            lorentz_feedback=False,
            stop_time=0.2,
            timestep=0.001,
            output_dt=0.05,
            snapshot_count=2,
        )
        norm = continuous_normalization()
        config.update(norm)
        config["initial_omega"] = lambda theta, r: reference_fields(theta, r)["omega"]
        result = run_spectral(config)
        last = result["snapshots"][-1]
        f = last["fields"]
        rr, tt = np.meshgrid(f["radius"], f["theta"])
        exact = last["time"] * reference_fields(tt, rr)["source"]
        records.append(
            dict(
                kind="winding",
                ntheta=n,
                nr=3 * n // 2,
                error=error_norm(last, exact),
                div_max=divergence_residual(last),
                integrals=independent_integrals(last),
            )
        )
        config.update(field_strength=0.0, poloidal_scale=0.0, eta=0.01, stop_time=0.2)
        config["initial_omega"] = lambda theta, r: np.zeros(
            np.broadcast_shapes(np.shape(theta), np.shape(r))
        )
        config["initial_q"] = lambda theta, r: np.broadcast_to(
            decay_mode(r)[1], np.broadcast_shapes(np.shape(theta), np.shape(r))
        ).copy()
        result = run_spectral(config)
        last = result["snapshots"][-1]
        rr, tt = np.meshgrid(last["fields"]["radius"], last["fields"]["theta"])
        k, q0 = decay_mode(rr)
        exact = q0 * np.exp(-0.01 * k * k * last["time"])
        records.append(
            dict(
                kind="resistive_decay",
                ntheta=n,
                nr=3 * n // 2,
                k=k,
                error=error_norm(last, exact),
                integrals=independent_integrals(last),
            )
        )
    return records


def compare_native_fields(grid, omega, q, spectral_snapshot):
    """Keep regularized and physical maxima, with fixed physical interior masks."""
    f = spectral_snapshot["fields"]
    ref_o = tensor_evaluate(
        f["theta"], f["radius"], f["omega"], np.cos(grid.theta), grid.radius
    )
    ref_q = tensor_evaluate(
        f["theta"], f["radius"], f["q"], np.cos(grid.theta), grid.radius
    )
    interior = (
        (grid.theta2d > 0.2)
        & (grid.theta2d < np.pi - 0.2)
        & (grid.radius2d > grid.r_inner + 0.04)
        & (grid.radius2d < grid.r_outer - 0.04)
    )
    result = {}
    for key, value, ref in [("omega", omega, ref_o), ("q", q, ref_q)]:
        diff = value - ref
        location = np.unravel_index(np.argmax(abs(diff)), diff.shape)
        result[key + "_relative_weighted_L2"] = float(
            np.sqrt(np.sum(grid.mass_s2 * diff**2) / np.sum(grid.mass_s2 * ref**2))
        )
        result[key + "_max_abs"] = float(np.max(abs(diff)))
        result[key + "_physical_component_max_abs"] = float(np.max(abs(grid.s * diff)))
        result[key + "_interior_max_abs"] = float(np.max(abs(diff[interior])))
        result[key + "_boundary_zone_max_abs"] = float(np.max(abs(diff[~interior])))
        result[key + "_physical_interior_max_abs"] = float(
            np.max(abs((grid.s * diff)[interior]))
        )
        result[key + "_physical_boundary_zone_max_abs"] = float(
            np.max(abs((grid.s * diff)[~interior]))
        )
        result[key + "_max_location"] = dict(
            theta=float(grid.theta2d[location]),
            radius=float(grid.radius2d[location]),
            reference_extrapolated=bool(
                grid.theta2d[location] < min(f["theta"])
                or grid.theta2d[location] > max(f["theta"])
                or grid.radius2d[location] < min(f["radius"])
                or grid.radius2d[location] > max(f["radius"])
            ),
        )
    return result, ref_o, ref_q


def matched_fd_studies(run_spectral, quick=False, stop_time=None, reference_dir=None):
    """Hold continuous initial data fixed while comparing independent spatial methods."""
    from nsinduct import shell

    spectral_sizes = [12, 16] if quick else [24, 32]
    records = []
    spectral_runs = []
    stop = (1.0 if quick else 10.0) if stop_time is None else float(stop_time)
    norm = continuous_normalization()
    for n in spectral_sizes:
        config = dict(
            ntheta=n,
            nr=3 * n // 2,
            field_strength=0.07,
            eta=2e-5,
            nu=0.0,
            lorentz_feedback=True,
            stop_time=stop,
            timestep=0.005 if n == 32 else 0.01,
            output_dt=0.1,
            snapshot_count=3,
            **norm,
        )
        config["initial_omega"] = lambda theta, r: reference_fields(theta, r)["omega"]
        config["max_wall_seconds"] = 300.0
        save_dir = reference_dir if n == spectral_sizes[-1] else None
        result = run_spectral(config, save_dir)
        snapshots = result["snapshots"]
        independent = [independent_integrals(s) for s in snapshots]
        spectral_runs.append(result)
        records.append(
            dict(
                kind="spectral_resolution",
                ntheta=n,
                nr=3 * n // 2,
                time=stop,
                summary=result["summary"],
                independent_diagnostics=independent,
                div_max=divergence_residual(snapshots[-1]),
            )
        )
    reference = spectral_runs[-1]["snapshots"][-1]
    for n in [16, 24] if quick else [24, 32, 48]:
        g = shell.make_grid(dict(ntheta=n, nr=3 * n // 2))
        exact = reference_fields(g.theta2d, g.radius2d)
        omega = exact["omega"].copy()
        q = np.zeros_like(omega)
        steps = max(1, int(np.ceil(stop / 0.005)))
        dt = stop / steps
        initial = shell.energies(g, omega, q, exact["br"], exact["btheta"])
        initial_j = float(np.sum(g.mass_s2 * omega))
        for _ in range(steps):
            omega, q = shell.rk4_step(
                g, omega, q, exact["br"], exact["btheta"], dt, 2e-5, 0.0, True
            )
        energy = shell.energies(g, omega, q, exact["br"], exact["btheta"])
        errors, _, _ = compare_native_fields(g, omega, q, reference)
        records.append(
            dict(
                kind="matched_fd",
                ntheta=n,
                nr=3 * n // 2,
                time=stop,
                errors=errors,
                E_kin=energy["kinetic"],
                E_tor=energy["magnetic_toroidal"],
                shear_fraction=float(np.sqrt(energy["kinetic"] / initial["kinetic"])),
                J=float(np.sum(g.mass_s2 * omega)),
                initial_J=initial_j,
                delta_J=float(np.sum(g.mass_s2 * omega)) - initial_j,
                initial_E_kin=initial["kinetic"],
                reference="highest spectral grid",
            )
        )
    time_grid = spectral_sizes[-1] if quick else 24
    config = dict(
        ntheta=time_grid,
        nr=3 * time_grid // 2,
        field_strength=0.07,
        eta=2e-5,
        nu=0.0,
        lorentz_feedback=True,
        stop_time=stop,
        output_dt=0.1,
        snapshot_count=2,
        **norm,
    )
    config["initial_omega"] = lambda theta, r: reference_fields(theta, r)["omega"]
    temporal = {0.01: spectral_runs[spectral_sizes.index(time_grid)]}
    for dt in [0.04, 0.02, 0.005]:
        temporal[dt] = run_spectral(config | dict(timestep=dt, max_wall_seconds=300.0))
    tf = temporal[0.005]["snapshots"][-1]["fields"]
    for dt in [0.04, 0.02, 0.01]:
        last = temporal[dt]["snapshots"][-1]
        f = last["fields"]
        qref = tensor_evaluate(
            tf["theta"], tf["radius"], tf["q"], np.cos(f["theta"]), f["radius"]
        )
        records.append(
            dict(
                kind="spectral_timestep",
                timestep=dt,
                reference_timestep=0.005,
                relative_q_L2=error_norm(last, qref),
                integrals=independent_integrals(last),
            )
        )
    return records


def braking_study(run_spectral, output_dir):
    """Measure sampled half-energy times and first peaks with fixed physical eta."""
    records = []
    for strength in [0.035, 0.07, 0.14]:
        norm = continuous_normalization(strength)
        config = dict(
            ntheta=24,
            nr=36,
            field_strength=strength,
            eta=2e-5,
            nu=0.0,
            lorentz_feedback=True,
            stop_time=0.98 / strength,
            timestep=0.0007 / strength,
            output_dt=0.1,
            snapshot_count=3,
            max_wall_seconds=300.0,
            **norm,
        )
        config["initial_omega"] = lambda theta, r: reference_fields(theta, r)["omega"]
        result = run_spectral(config, output_dir / ("braking_" + str(strength)))
        records.append(
            dict(
                kind="braking",
                field_strength=strength,
                summary=result["summary"],
                independent_diagnostics=[
                    independent_integrals(s) for s in result["snapshots"]
                ],
                runtime=result["metadata"].get("elapsed_seconds"),
                peak_memory_bytes=result["metadata"].get("peak_memory_bytes"),
            )
        )
    return records


def diagnose_saved_reference(reference_dir, out):
    """Recalculate FD comparisons and figures against an existing canonical spectral run."""
    import csv
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nsinduct import shell

    reference_dir = Path(reference_dir)
    metadata = json.loads((reference_dir / "metadata.json").read_text())
    norm = continuous_normalization()
    if any(
        not np.isclose(metadata["normalization"][k], value, rtol=1e-11)
        for k, value in norm.items()
    ):
        raise ValueError(
            "Reference must use the canonical Bp=.07 continuum normalization"
        )
    with h5py.File(reference_dir / "snapshots.h5") as store:
        snapshots = [
            dict(time=float(g.attrs["time"]), fields={k: g[k][...] for k in g})
            for g in store.values()
        ]
    reference = snapshots[-1]
    f = reference["fields"]
    stop = reference["time"]
    records = []
    curves = []
    for n in [24, 32, 48]:
        g = shell.make_grid(dict(ntheta=n, nr=3 * n // 2))
        initial = reference_fields(g.theta2d, g.radius2d)
        omega = initial["omega"].copy()
        q = np.zeros_like(omega)
        e0 = shell.energies(g, omega, q, initial["br"], initial["btheta"])["kinetic"]
        j0 = float(np.sum(g.mass_s2 * omega))
        count = max(1, int(np.ceil(stop / 0.005)))
        dt = stop / count
        curve = []
        for step in range(count + 1):
            if step % 20 == 0 or step == count:
                e = shell.energies(g, omega, q, initial["br"], initial["btheta"])
                curve.append(
                    [step * dt, e["kinetic"] / e0, e["magnetic_toroidal"] / e0]
                )
            if step < count:
                omega, q = shell.rk4_step(
                    g,
                    omega,
                    q,
                    initial["br"],
                    initial["btheta"],
                    dt,
                    metadata["eta"],
                    metadata["nu"],
                    True,
                )
        errors, omega_ref, q_ref = compare_native_fields(g, omega, q, reference)
        records.append(
            dict(
                ntheta=n,
                nr=3 * n // 2,
                time=stop,
                errors=errors,
                initial_J=j0,
                delta_J=float(np.sum(g.mass_s2 * omega)) - j0,
            )
        )
        curves.append((n, np.asarray(curve)))
        if n == 48:
            np.savez_compressed(
                out / "fd48_vs_spectral_t10.npz",
                theta=g.theta,
                radius=g.radius,
                omega=omega,
                q=q,
                omega_reference=omega_ref,
                q_reference=q_ref,
            )
    report = dict(
        spectral_independent=[independent_integrals(x) for x in snapshots],
        records=records,
    )
    (out / "physical_component_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (reference_dir / "diagnostics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = [float(x["time"]) for x in rows]
    spec_e0 = float(rows[0]["E_kin"])
    label = rf"Dedalus ${len(f['theta'])}\times{len(f['radius'])}$"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for j, key in enumerate(["E_kin", "E_tor"]):
        axes[j].plot(
            times, [float(x[key]) / spec_e0 for x in rows], label=label, color="black"
        )
        for n, curve in curves:
            axes[j].plot(
                curve[:, 0], curve[:, j + 1], "--", label=rf"FD ${n}\times{3 * n // 2}$"
            )
        axes[j].set(xlabel=r"$t$", ylabel=[r"$E_K/E_K(0)$", r"$E_T/E_K(0)$"][j])
        axes[j].legend(fontsize=8)
    fig.savefig(out / "matched_energy.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ns = [x["ntheta"] for x in records]
    for ax, key in zip(axes, ["omega", "q"]):
        for suffix, marker, title in [
            ("_relative_weighted_L2", "o-", "weighted field L2"),
            ("_max_abs", "s--", "regularized variable max"),
            ("_physical_component_max_abs", "^--", "physical component max"),
        ]:
            ax.loglog(
                ns, [x["errors"][key + suffix] for x in records], marker, label=title
            )
        ax.set(
            xlabel=r"$N_\theta$ (FD)",
            ylabel="error against " + label,
            title={"omega": r"$\Omega$", "q": r"$q$"}[key],
        )
        ax.legend(fontsize=8)
    fig.savefig(out / "field_norm_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    rr, tt = np.meshgrid(f["radius"], f["theta"])
    ss = rr * np.sin(tt)
    zz = rr * np.cos(tt)
    limit = max(np.max(abs(f["B_phi"])), np.max(abs(g.s * q)))
    levels = np.linspace(-limit, limit, 31)
    a = axes[0].contourf(zz, ss, f["B_phi"], levels=levels, cmap="RdBu_r")
    axes[1].contourf(g.z, g.s, g.s * q, levels=levels, cmap="RdBu_r")
    difference = g.s * (q - q_ref)
    limit = np.max(abs(difference))
    c = axes[2].contourf(
        g.z, g.s, difference, levels=np.linspace(-limit, limit, 31), cmap="RdBu_r"
    )
    for ax, title in zip(
        axes,
        [
            label + r": $B_\phi$",
            r"FD $48\times72$: $B_\phi$",
            r"FD $-$ Dedalus: $B_\phi$",
        ],
    ):
        ax.set(xlabel=r"$z$", ylabel=r"$s$", title=title, aspect="equal")
    fig.colorbar(a, ax=axes[:2], label=r"$B_\phi$")
    fig.colorbar(c, ax=axes[2], label=r"$\Delta B_\phi$")
    fig.savefig(out / "matched_toroidal_fields.png", dpi=150)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--matched",
        action="store_true",
        help="Also run matched coupled FD/refinement comparisons",
    )
    parser.add_argument(
        "--stop-time",
        type=float,
        help="Common final time for matched coupled comparisons",
    )
    parser.add_argument("--skip-analytic", action="store_true")
    parser.add_argument(
        "--braking",
        action="store_true",
        help="Run three seed strengths through their first energy maxima",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Diagnose and plot a saved canonical spectral reference without rerunning Dedalus",
    )
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Choose output outside source tree")
    out.mkdir(parents=True, exist_ok=True)
    if args.stop_time is not None and (
        not np.isfinite(args.stop_time) or args.stop_time <= 0
    ):
        parser.error("--stop-time must be finite and positive")
    if args.reference_dir:
        report = diagnose_saved_reference(args.reference_dir, out)
        print(json.dumps(report, indent=2))
        return
    from nsinduct.dedalus_backend import run_case

    records = [] if args.skip_analytic else analytic_experiments(run_case, args.quick)
    if args.matched:
        matched = matched_fd_studies(
            run_case, args.quick, args.stop_time, out / "matched_reference"
        )
        records.extend(matched)
    if args.braking:
        records.extend(braking_study(run_case, out))
    report = dict(records=records)
    (out / "independent_analytic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
