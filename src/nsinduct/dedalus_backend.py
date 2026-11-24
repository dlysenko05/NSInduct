"""Axisymmetric toroidal-vector ShellBasis IVPs, independently of the FD solver.

U=s Omega e_phi and T=s q e_phi are proper spin-1 vector fields. The retained
force curl(T) cross P is toroidal; omitted poloidal forces are not evolved.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre


def normalization(config):
    """Continuous-volume reference constants; optional explicit shared overrides."""
    ri = float(config.get("r_inner", 0.3))
    ro = float(config.get("r_outer", 1.0))
    x, w = roots_legendre(160)
    r = ri + (x + 1) * (ro - ri) / 2
    mu = x[:, None]
    rr = r[None, :]
    vol_weights = 2 * np.pi * w[:, None] * w[None, :] * (ro - ri) / 2 * rr**2
    mass = vol_weights * rr**2 * (1 - mu**2)
    h = ((rr - ri) * (ro - rr)) ** 2
    hp = 2 * (rr - ri) * (ro - rr) * (ri + ro - 2 * rr)
    b2 = (2 * h * mu / rr**2) ** 2 + (hp * np.sqrt(1 - mu**2) / rr) ** 2
    bp_unit_rms = np.sqrt(np.sum(vol_weights * b2) / np.sum(vol_weights))
    raw = (
        (
            1
            + float(config.get("radial_contrast", 0.18))
            * np.cos(2 * np.pi * (rr - ri) / (ro - ri))
        )
        * (3 * mu**2 - 1)
        / 2
    )
    mean = np.sum(mass * raw) / np.sum(mass)
    amp = np.sqrt(np.sum(mass * (raw - mean) ** 2) / np.sum(mass))
    return dict(
        poloidal_scale=float(
            config.get(
                "poloidal_scale",
                float(config.get("field_strength", 0.07)) / bp_unit_rms,
            )
        ),
        omega_mean=float(config.get("omega_mean", mean)),
        omega_amplitude=float(config.get("omega_amplitude", amp)),
        bp_unit_rms=float(bp_unit_rms),
    )


def run_case(config, output_dir=None):
    """Evolve the shell in serial and return snapshots on sorted native nodes.

    Initial-condition callbacks take the broadcast colatitude and radius grids.
    """
    import dedalus
    import dedalus.public as d3
    from mpi4py import MPI

    if MPI.COMM_WORLD.size != 1:
        raise ValueError("This verified backend currently supports serial MPI only")
    ntheta = int(config.get("ntheta", 16))
    nr = int(config.get("nr", 24))
    ri = float(config.get("r_inner", 0.3))
    ro = float(config.get("r_outer", 1.0))
    eta = float(config.get("eta", 2e-5))
    nu = float(config.get("nu", 0.0))
    timestep = float(config.get("timestep", 0.005))
    stop = float(config.get("stop_time", 1.0))
    output_dt = float(config.get("output_dt", 0.1))
    snap_count = int(config.get("snapshot_count", 3))
    feedback = bool(config.get("lorentz_feedback", True))
    wall_limit = float(config.get("max_wall_seconds", 120.0))
    iteration_limit = int(config.get("max_iterations", 200000))
    memory_limit = float(config.get("max_memory_mb", 2048.0)) * 1024**2
    if ntheta > 64 or nr > 96:
        raise ValueError("Dedalus runs require ntheta<=64 and nr<=96")
    if ntheta < 8 or nr < 8 or not 0 < ri < ro:
        raise ValueError("Require resolutions>=8 and 0<ri<ro")
    if (
        not np.all(np.isfinite([eta, nu, timestep, stop, output_dt]))
        or eta < 0
        or nu < 0
        or timestep <= 0
        or stop < 0
        or output_dt <= 0
        or snap_count < 2
    ):
        raise ValueError("Invalid diffusion, time or snapshot parameters")
    norm = normalization(config)
    coords = d3.SphericalCoordinates("phi", "theta", "r")
    dist = d3.Distributor(coords, dtype=np.float64)
    basis = d3.ShellBasis(
        coords, shape=(1, ntheta, nr), radii=(ri, ro), dealias=1.5, dtype=np.float64
    )
    _, theta, radius = dist.local_grids(basis)
    tt, rr = np.broadcast_arrays(theta[0], radius[0])
    ss = rr * np.sin(tt)
    U = dist.VectorField(coords, name="U", bases=basis)
    T = dist.VectorField(coords, name="T", bases=basis)
    P = dist.VectorField(coords, name="P", bases=basis)
    K = dist.VectorField(coords, name="K", bases=basis)
    h = ((radius - ri) * (ro - radius)) ** 2
    hp = 2 * (radius - ri) * (ro - radius) * (ri + ro - 2 * radius)
    P["g"][1] = -norm["poloidal_scale"] * hp * np.sin(theta) / radius
    P["g"][2] = norm["poloidal_scale"] * 2 * h * np.cos(theta) / radius**2
    native_br = P["g"][2].copy()
    native_btheta = P["g"][1].copy()
    K["g"][0] = radius * np.sin(theta)
    raw = (
        (
            1
            + float(config.get("radial_contrast", 0.18))
            * np.cos(2 * np.pi * (rr - ri) / (ro - ri))
        )
        * (3 * np.cos(tt) ** 2 - 1)
        / 2
    )
    omega0 = (
        float(config.get("delta_omega", 1.0))
        * (raw - norm["omega_mean"])
        / norm["omega_amplitude"]
    )
    if "initial_omega" in config:
        omega0 = np.broadcast_to(config["initial_omega"](tt, rr), ss.shape)
    q0 = np.zeros_like(ss)
    if "initial_q" in config:
        q0 = np.broadcast_to(config["initial_q"](tt, rr), ss.shape)
    if eta and "initial_q" in config:
        for wall in [ri, ro]:
            boundary = np.asarray(config["initial_q"](tt, np.full_like(rr, wall)))
            if np.max(abs(boundary)) > 1e-10 * max(1.0, float(np.max(abs(q0)))):
                raise ValueError("Resistive initial_q must vanish at both radial walls")
    U["g"][0] = ss * omega0
    T["g"][0] = ss * q0
    variables = [U, T]
    surface = basis.outer_surface
    lift_basis = basis.derivative_basis(2)

    def lift(field, n=-1):
        return d3.Lift(field, lift_basis, n)

    if eta:
        tau_T1 = dist.VectorField(coords, name="tau_T1", bases=surface)
        tau_T2 = dist.VectorField(coords, name="tau_T2", bases=surface)
        variables.extend([tau_T1, tau_T2])
    if nu:
        tau_U1 = dist.VectorField(coords, name="tau_U1", bases=surface)
        tau_U2 = dist.VectorField(coords, name="tau_U2", bases=surface)
        du_wall_inner = d3.radial(d3.grad(U)(r=ri), index=0)
        du_wall_outer = d3.radial(d3.grad(U)(r=ro), index=0)
        variables.extend([tau_U1, tau_U2])
    force = d3.cross(d3.curl(T), P) if feedback else 0 * U
    source = d3.curl(d3.cross(U, P))
    problem = d3.IVP(variables, namespace=locals())
    if nu:
        problem.add_equation(
            "dt(U) - nu*lap(U) + lift(tau_U1,-1) + lift(tau_U2,-2) = force"
        )
        problem.add_equation("ri*du_wall_inner-U(r=ri)=0")
        problem.add_equation("ro*du_wall_outer-U(r=ro)=0")
    else:
        problem.add_equation("dt(U)=force")
    if eta:
        problem.add_equation(
            "dt(T) - eta*lap(T) + lift(tau_T1,-1) + lift(tau_T2,-2) = source"
        )
        problem.add_equation("T(r=ri)=0")
        problem.add_equation("T(r=ro)=0")
    else:
        problem.add_equation("dt(T)=source")
    started = time.perf_counter()
    solver = problem.build_solver(d3.RK443)
    build_seconds = time.perf_counter() - started

    def peak_memory():
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage if sys.platform == "darwin" else usage * 1024)

    def integral(expression):
        return float(np.asarray(d3.integ(expression).evaluate()["g"]).ravel()[0])

    volume = 4 * np.pi * (ro**3 - ri**3) / 3
    pol_energy = 0.5 * integral(P @ P)
    div_p_rms = np.sqrt(max(0, integral(d3.div(P) ** 2) / volume))
    tnodes = np.asarray(theta).ravel()
    rnodes = np.asarray(radius).ravel()
    ti = np.argsort(tnodes)
    ri_order = np.argsort(rnodes)
    theta_sorted = tnodes[ti]
    radius_sorted = rnodes[ri_order]
    r2, t2 = np.meshgrid(radius_sorted, theta_sorted)
    s2 = r2 * np.sin(t2)

    def sorted_scalar(array):
        return np.asarray(array)[0][np.ix_(ti, ri_order)].copy()

    def sample_fields():
        for f in [U, T]:
            f.change_scales(1)
        uphi = sorted_scalar(U["g"][0])
        bphi = sorted_scalar(T["g"][0])
        br = sorted_scalar(native_br)
        bt = sorted_scalar(native_btheta)
        psi = norm["poloidal_scale"] * ((r2 - ri) * (ro - r2)) ** 2 * np.sin(t2) ** 2
        return dict(
            theta=theta_sorted.copy(),
            radius=radius_sorted.copy(),
            s=s2,
            z=r2 * np.cos(t2),
            omega=uphi / s2,
            q=bphi / s2,
            B_phi=bphi,
            u_phi=uphi,
            B_r=br,
            B_theta=bt,
            psi=psi,
        )

    rows = []
    snapshots = []
    loss_ohm = 0.0
    loss_visc = 0.0
    driving_work = 0.0

    def diagnostic(t):
        e_kin = 0.5 * integral(U @ U)
        e_tor = 0.5 * integral(T @ T)
        j = integral(K @ U)
        power = integral(T @ source)
        work = integral(U @ d3.cross(d3.curl(T), P)) if feedback else 0.0
        ohm = eta * integral(d3.curl(T) @ d3.curl(T)) if eta else 0.0
        if nu:
            gradient = d3.grad(U)
            strain = (gradient + d3.transpose(gradient)) / 2
            visc = 2 * nu * integral(d3.trace(strain @ strain))
        else:
            visc = 0.0
        fields = sample_fields()
        physical_viscous_wall = 0.0
        if nu:
            inner_stress = d3.radial(d3.grad(U)(r=ri), index=0) - U(r=ri) / ri
            outer_stress = d3.radial(d3.grad(U)(r=ro), index=0) - U(r=ro) / ro
            physical_viscous_wall = max(
                float(np.max(abs(inner_stress.evaluate()["g"]))),
                float(np.max(abs(outer_stress.evaluate()["g"]))),
            )
        wall = max(
            float(np.max(abs(T(r=ri).evaluate()["g"]))),
            float(np.max(abs(T(r=ro).evaluate()["g"]))),
        )
        row = dict(
            time=float(t),
            E_kin=e_kin,
            E_tor=e_tor,
            E_pol=pol_energy,
            E_total=e_kin + e_tor + pol_energy,
            angular_momentum_total=j,
            D_ohmic=ohm,
            D_viscous=visc,
            magnetic_winding_power=power,
            lorentz_work_rate=work,
            exchange_residual=power + work if feedback else 0.0,
            prescribed_flow_power=0.0 if feedback else power,
            cum_ohmic=loss_ohm,
            cum_viscous=loss_visc,
            cum_prescribed_flow_work=driving_work,
            max_abs_B_phi=float(np.max(abs(fields["B_phi"]))),
            min_omega=float(np.min(fields["omega"])),
            max_omega=float(np.max(fields["omega"])),
            shear_amplitude=np.sqrt(max(0, 2 * e_kin / integral(K @ K))),
            toroidal_wall_residual=wall,
            physical_viscous_wall_residual=physical_viscous_wall,
            div_Bp_rms=div_p_rms,
            div_T_rms=np.sqrt(max(0, integral(d3.div(T) ** 2) / volume)),
            div_U_rms=np.sqrt(max(0, integral(d3.div(U) ** 2) / volume)),
            max_meridional_U=float(np.max(abs(U["g"][1:]))),
            max_meridional_T=float(np.max(abs(T["g"][1:]))),
        )
        return row, fields

    output_times = np.unique(
        np.r_[np.arange(0, stop + output_dt * 0.25, output_dt), stop]
    )
    output_times = output_times[output_times <= stop + 1e-12]
    snapshot_times = np.linspace(0, stop, snap_count)
    ordered_targets = np.sort(np.r_[output_times, snapshot_times])
    targets = [ordered_targets[0]]
    for target in ordered_targets[1:]:
        if target - targets[-1] > 1e-12:
            targets.append(target)
    row, fields = diagnostic(0)
    e_initial = row["E_total"]
    prev_rates = (row["D_ohmic"], row["D_viscous"], row["prescribed_flow_power"])
    for target in targets:
        while solver.sim_time < float(target) - 1e-12:
            step = min(timestep, float(target) - solver.sim_time)
            if (
                time.perf_counter() - started > wall_limit
                or solver.iteration >= iteration_limit
                or peak_memory() > memory_limit
            ):
                raise RuntimeError(
                    "Dedalus experiment exceeded configured wall-time, iteration or peak-memory budget"
                )
            solver.step(step)
            # Per-step quadrature isolates time integration from sparse saved outputs.
            ohm = eta * integral(d3.curl(T) @ d3.curl(T)) if eta else 0.0
            if nu:
                gu = d3.grad(U)
                strain = (gu + d3.transpose(gu)) / 2
                visc = 2 * nu * integral(d3.trace(strain @ strain))
            else:
                visc = 0.0
            power = integral(T @ source) if not feedback else 0.0
            if not np.all(np.isfinite([ohm, visc, power])):
                raise FloatingPointError(
                    f"Nonfinite Dedalus rate at t={solver.sim_time:g}, dt={step:g}, shape={(ntheta, nr)}"
                )
            loss_ohm += step * (prev_rates[0] + ohm) / 2
            loss_visc += step * (prev_rates[1] + visc) / 2
            driving_work += step * (prev_rates[2] + power) / 2
            prev_rates = (ohm, visc, power)
        row, fields = diagnostic(float(target))
        if not np.isfinite(row["E_total"]):
            raise FloatingPointError(
                f"Nonfinite Dedalus state at t={solver.sim_time:g}, dt={step if solver.iteration else 0:g}, shape={(ntheta, nr)}"
            )
        row["cumulative_energy_residual"] = (
            row["E_total"] - e_initial + loss_ohm + loss_visc - driving_work
        ) / max(e_initial, 1e-300)
        if np.any(np.isclose(output_times, target, rtol=0, atol=1e-12)):
            rows.append(row)
        if np.any(np.isclose(snapshot_times, target, rtol=0, atol=1e-12)):
            snapshots.append(dict(time=float(target), fields=fields))
    metadata = dict(
        backend="dedalus_shellbasis_toroidal_vectors",
        dedalus_version=dedalus.__version__,
        shape=[1, ntheta, nr],
        dealias=1.5,
        timestepper="RK443",
        normalization=norm,
        eta=eta,
        nu=nu,
        lorentz_feedback=feedback,
        build_seconds=build_seconds,
        elapsed_seconds=time.perf_counter() - started,
        peak_memory_bytes=peak_memory(),
        serial_mpi=True,
        budgets=dict(
            max_wall_seconds=wall_limit,
            max_iterations=iteration_limit,
            max_memory_bytes=memory_limit,
        ),
        toroidal_B_BC="zero at both radial walls"
        if eta
        else "compatible initial condition; no diffusive tau",
        velocity_BC="stress-free d_r(U/r)=0"
        if nu
        else "ideal characteristic coupling, P vanishes at walls",
    )
    peak = max(rows, key=lambda x: x["E_tor"])
    e_kin_initial = rows[0]["E_kin"]
    half = next(
        (r["time"] for r in rows if 0 < e_kin_initial and r["E_kin"] <= 0.5 * e_kin_initial),
        None,
    )
    summary = dict(
        peak_E_tor=peak["E_tor"],
        time_peak_E_tor=peak["time"],
        braking_time_half_Ekin=half,
        max_cumulative_energy_residual=max(
            abs(r["cumulative_energy_residual"]) for r in rows
        ),
        max_toroidal_wall_residual=max(r["toroidal_wall_residual"] for r in rows),
    )
    if output_dir is not None:
        import h5py

        from .shell import write_csv

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        write_csv(path / "diagnostics.csv", rows)
        with h5py.File(path / "snapshots.h5", "w") as store:
            for index, snap in enumerate(snapshots):
                group = store.create_group(f"snap_{index:03d}")
                group.attrs["time"] = snap["time"]
                for key, val in snap["fields"].items():
                    group.create_dataset(key, data=val)
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        saved_config = {
            key: val if not callable(val) else getattr(val, "__name__", "callback")
            for key, val in config.items()
        }
        (path / "config.json").write_text(json.dumps(saved_config, indent=2))
    return dict(rows=rows, snapshots=snapshots, summary=summary, metadata=metadata)
