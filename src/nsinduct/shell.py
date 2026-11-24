"""Finite differences for axisymmetric torsional MHD in a spherical shell.

The state is (Omega, B_phi/(r sin(theta))). Face-integrated Maxwell stresses
give a conservative torque. Winding is the negative weighted adjoint of that
torque, so ideal energy exchange cancels on the grid.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

MU0 = 1.0
RHO0 = 1.0


@dataclass(frozen=True)
class ShellGrid:
    """Cell-centered axisymmetric spherical-shell grid."""

    ntheta: int
    nr: int
    r_inner: float
    r_outer: float
    theta: np.ndarray
    radius: np.ndarray
    theta2d: np.ndarray
    radius2d: np.ndarray
    s: np.ndarray
    z: np.ndarray
    weights: np.ndarray
    mass_s2: np.ndarray
    dr: float
    dtheta: float

    @property
    def shell_length(self) -> float:
        return self.r_outer - self.r_inner

    @property
    def volume(self) -> float:
        return float(np.sum(self.weights))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML configuration must be a mapping: {path}")
    return data


def dump_yaml(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return output
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output


def read_csv(path: str | Path) -> list[dict[str, float | str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float | str]] = []
        for row in reader:
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = ""
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def make_grid(config: dict[str, Any]) -> ShellGrid:
    ntheta = int(config.get("ntheta", 48))
    nr = int(config.get("nr", 72))
    r_inner = float(config.get("r_inner", 0.3))
    r_outer = float(config.get("r_outer", 1.0))
    if ntheta < 8 or nr < 8:
        raise ValueError("ntheta and nr must both be at least 8")
    if not (0.0 < r_inner < r_outer):
        raise ValueError("require 0 < r_inner < r_outer")

    dr = (r_outer - r_inner) / nr
    dtheta = math.pi / ntheta
    radius = r_inner + (np.arange(nr, dtype=float) + 0.5) * dr
    theta = (np.arange(ntheta, dtype=float) + 0.5) * dtheta
    radius2d, theta2d = np.meshgrid(radius, theta)
    s = radius2d * np.sin(theta2d)
    z = radius2d * np.cos(theta2d)
    radial_faces = r_inner + np.arange(nr + 1) * dr
    theta_faces = np.arange(ntheta + 1) * dtheta
    angular_mass = np.diff(np.cos(theta_faces) ** 3 / 3 - np.cos(theta_faces))
    radial_mass = np.diff(radial_faces**5) / 5
    mass_s2 = 2 * math.pi * angular_mass[:, None] * radial_mass[None, :]
    weights = mass_s2 / s**2
    return ShellGrid(
        ntheta=ntheta,
        nr=nr,
        r_inner=r_inner,
        r_outer=r_outer,
        theta=theta,
        radius=radius,
        theta2d=theta2d,
        radius2d=radius2d,
        s=s,
        z=z,
        weights=weights,
        mass_s2=mass_s2,
        dr=dr,
        dtheta=dtheta,
    )


def _ghost(
    field: np.ndarray, radial: str = "neumann", theta: str = "neumann"
) -> np.ndarray:
    if radial not in {"neumann", "dirichlet"} or theta not in {"neumann", "dirichlet"}:
        raise ValueError("boundary type must be 'neumann' or 'dirichlet'")
    padded = np.empty((field.shape[0] + 2, field.shape[1] + 2), dtype=float)
    padded[1:-1, 1:-1] = field
    if radial == "neumann":
        padded[1:-1, 0] = field[:, 0]
        padded[1:-1, -1] = field[:, -1]
    else:
        padded[1:-1, 0] = -field[:, 0]
        padded[1:-1, -1] = -field[:, -1]
    if theta == "neumann":
        padded[0, 1:-1] = field[0, :]
        padded[-1, 1:-1] = field[-1, :]
    else:
        padded[0, 1:-1] = -field[0, :]
        padded[-1, 1:-1] = -field[-1, :]
    padded[0, 0] = padded[0, 1]
    padded[0, -1] = padded[0, -2]
    padded[-1, 0] = padded[-1, 1]
    padded[-1, -1] = padded[-1, -2]
    return padded


def d_dr(
    grid: ShellGrid, field: np.ndarray, radial: str = "neumann", theta: str = "neumann"
) -> np.ndarray:
    padded = _ghost(field, radial, theta)
    return (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * grid.dr)


def d_dtheta(
    grid: ShellGrid, field: np.ndarray, radial: str = "neumann", theta: str = "neumann"
) -> np.ndarray:
    padded = _ghost(field, radial, theta)
    return (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * grid.dtheta)


def volume_integral(grid: ShellGrid, field: np.ndarray) -> float:
    return float(np.sum(grid.weights * field))


def volume_rms(grid: ShellGrid, field: np.ndarray) -> float:
    return math.sqrt(volume_integral(grid, field**2) / grid.volume)


def weighted_shear_amplitude(grid: ShellGrid, omega: np.ndarray) -> float:
    denom = float(np.sum(grid.mass_s2))
    return math.sqrt(float(np.sum(grid.mass_s2 * omega**2)) / denom)


def poloidal_field(
    grid: ShellGrid, field_strength: float
) -> dict[str, np.ndarray | float]:
    """Return a smooth shell-confined dipolar poloidal field.

    The flux function is psi = C (r-Ri)^2 (Ro-r)^2 sin^2(theta).  The double
    radial zeros make B_r and B_theta vanish at both shell boundaries.
    """

    r = grid.radius2d
    theta = grid.theta2d
    h = ((r - grid.r_inner) * (grid.r_outer - r)) ** 2
    hp = (
        2.0
        * (r - grid.r_inner)
        * (grid.r_outer - r)
        * (grid.r_inner + grid.r_outer - 2.0 * r)
    )
    psi = h * np.sin(theta) ** 2
    br = 2.0 * h * np.cos(theta) / r**2
    btheta = -hp * np.sin(theta) / r
    rms = volume_rms(grid, np.sqrt(br**2 + btheta**2))
    if rms <= 0.0:
        raise ValueError("poloidal field normalization failed")
    scale = float(field_strength) / rms
    br *= scale
    btheta *= scale
    psi *= scale
    return {
        "psi": psi,
        "br": br,
        "btheta": btheta,
        "bp_rms": volume_rms(grid, np.sqrt(br**2 + btheta**2)),
        "bp_max": float(np.max(np.sqrt(br**2 + btheta**2))),
    }


def initial_omega(grid: ShellGrid, config: dict[str, Any]) -> np.ndarray:
    delta_omega = float(config.get("delta_omega", 1.0))
    radial_contrast = float(config.get("radial_contrast", 0.18))
    x = (grid.radius2d - grid.r_inner) / grid.shell_length
    p2 = 0.5 * (3.0 * np.cos(grid.theta2d) ** 2 - 1.0)
    raw = (1.0 + radial_contrast * np.cos(2.0 * math.pi * x)) * p2

    # Remove solid-body angular momentum so E_kin tracks differential rotation.
    solid = float(np.sum(grid.mass_s2 * raw) / np.sum(grid.mass_s2))
    omega = raw - solid
    amp = weighted_shear_amplitude(grid, omega)
    if amp <= 0.0:
        raise ValueError("initial differential rotation has zero amplitude")
    return omega * (delta_omega / amp)


@lru_cache(maxsize=16)
def _face_geometry(ntheta, nr, r_inner, r_outer):
    """Exact flux-face integrals for the fixed quartic poloidal geometry."""
    dr = (r_outer - r_inner) / nr
    te = np.arange(ntheta + 1) * np.pi / ntheta
    re = r_inner + np.arange(nr + 1) * dr
    radius = (re[:-1] + re[1:]) / 2
    theta = (te[:-1] + te[1:]) / 2
    r, t = np.meshgrid(radius, theta)
    h = ((r - r_inner) * (r_outer - r)) ** 2
    hp = 2 * (r - r_inner) * (r_outer - r) * (r_inner + r_outer - 2 * r)
    raw_br = 2 * h * np.cos(t) / r**2
    raw_bt = -hp * np.sin(t) / r
    hf = ((re - r_inner) * (r_outer - re)) ** 2
    ar = (np.sin(te[1:]) ** 4 - np.sin(te[:-1]) ** 4) / 4
    cr = 2 * np.pi * 2 * ar[:, None] * hf[None, :] * re[None, :] ** 2
    nodes, weights = np.polynomial.legendre.leggauss(3)
    rr = radius[None, :] + dr / 2 * nodes[:, None]
    hp = 2 * (rr - r_inner) * (r_outer - rr) * (r_inner + r_outer - 2 * rr)
    integral = np.sum(weights[:, None] * rr**2 * hp, axis=0) * dr / 2
    ct = -2 * np.pi * np.sin(te[:, None]) ** 4 * integral[None, :]
    return raw_br, raw_bt, cr, ct


def _ideal_face_coefficients(g, br, bt):
    raw_br, raw_bt, cr, ct = _face_geometry(g.ntheta, g.nr, g.r_inner, g.r_outer)
    scale = np.sum(raw_br * br) / np.sum(raw_br**2)
    if not (
        np.allclose(br, scale * raw_br, rtol=1e-10, atol=1e-14)
        and np.allclose(bt, scale * raw_bt, rtol=1e-10, atol=1e-14)
    ):
        raise ValueError(
            "Ideal face operators require the confined quartic poloidal field"
        )
    return scale * cr, scale * ct


def winding_operator(g, br, bt, o):
    cr, ct = _ideal_face_coefficients(g, br, bt)
    out = np.zeros_like(o)
    fr = cr[:, 1:-1] * (o[:, 1:] - o[:, :-1]) / 2
    ft = ct[1:-1, :] * (o[1:, :] - o[:-1, :]) / 2
    out[:, :-1] += fr
    out[:, 1:] += fr
    out[:-1, :] += ft
    out[1:, :] += ft
    return out / g.mass_s2


def winding_adjoint(g, br, bt, weighted_q):
    cr, ct = _ideal_face_coefficients(g, br, bt)
    q = weighted_q / g.mass_s2
    fr = np.zeros((g.ntheta, g.nr + 1))
    ft = np.zeros((g.ntheta + 1, g.nr))
    fr[:, 1:-1] = cr[:, 1:-1] * (q[:, 1:] + q[:, :-1]) / 2
    ft[1:-1, :] = ct[1:-1, :] * (q[1:, :] + q[:-1, :]) / 2
    return -(fr[:, 1:] - fr[:, :-1] + ft[1:, :] - ft[:-1, :])


def curl_toroidal(
    grid: ShellGrid, toroidal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Curl of F e_phi for a regular toroidal scalar F = B_phi or u_phi."""

    jr = d_dtheta(grid, np.sin(grid.theta2d) * toroidal, "neumann", "neumann") / (
        grid.radius2d * np.sin(grid.theta2d)
    )
    jtheta = (
        -d_dr(grid, grid.radius2d * toroidal, "dirichlet", "neumann") / grid.radius2d
    )
    return jr, jtheta


def _scalar_diffusion(g, q, dirichlet):
    re = g.r_inner + np.arange(g.nr + 1) * g.dr
    te = np.arange(g.ntheta + 1) * g.dtheta
    ar = np.diff(np.cos(te) ** 3 / 3 - np.cos(te))
    rr = np.diff(re**3) / 3
    fr = np.zeros((g.ntheta, g.nr + 1))
    ft = np.zeros((g.ntheta + 1, g.nr))
    fr[:, 1:-1] = (q[:, 1:] - q[:, :-1]) / g.dr
    if dirichlet:
        fr[:, 0] = 2 * q[:, 0] / g.dr
        fr[:, -1] = -2 * q[:, -1] / g.dr
    fr *= 2 * np.pi * ar[:, None] * re[None, :] ** 4
    ft[1:-1, :] = (q[1:, :] - q[:-1, :]) / g.dtheta
    ft *= 2 * np.pi * np.sin(te[:, None]) ** 3 * rr[None, :]
    return (fr[:, 1:] - fr[:, :-1] + ft[1:, :] - ft[:-1, :]) / g.mass_s2


def negative_curl_curl_toroidal(g, f):
    jr, jt = curl_toroidal(g, f)
    return g.s * _scalar_diffusion(g, f / g.s, True), jr, jt


def negative_weighted_laplacian_omega(g, o):
    return _scalar_diffusion(g, o, False), d_dr(g, o), d_dtheta(g, o)


def rhs(
    grid: ShellGrid,
    omega: np.ndarray,
    q: np.ndarray,
    br: np.ndarray,
    btheta: np.ndarray,
    eta: float,
    nu: float,
    lorentz_feedback: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return time derivatives for Omega and q."""

    bphi = grid.s * q
    q_t = winding_operator(grid, br, btheta, omega)
    if eta != 0.0:
        diffusion_bphi, _jr, _jtheta = negative_curl_curl_toroidal(grid, bphi)
        q_t += eta * diffusion_bphi / grid.s

    if lorentz_feedback:
        omega_t = -winding_adjoint(grid, br, btheta, grid.mass_s2 * q) / grid.mass_s2
    else:
        omega_t = np.zeros_like(omega)
    if nu != 0.0:
        diffusion_omega, _omega_r, _omega_theta = negative_weighted_laplacian_omega(
            grid, omega
        )
        omega_t = omega_t + nu * diffusion_omega
    return omega_t, q_t


def rk4_step(
    grid: ShellGrid,
    omega: np.ndarray,
    q: np.ndarray,
    br: np.ndarray,
    btheta: np.ndarray,
    dt: float,
    eta: float,
    nu: float,
    lorentz_feedback: bool,
) -> tuple[np.ndarray, np.ndarray]:
    k1o, k1q = rhs(grid, omega, q, br, btheta, eta, nu, lorentz_feedback)
    k2o, k2q = rhs(
        grid,
        omega + 0.5 * dt * k1o,
        q + 0.5 * dt * k1q,
        br,
        btheta,
        eta,
        nu,
        lorentz_feedback,
    )
    k3o, k3q = rhs(
        grid,
        omega + 0.5 * dt * k2o,
        q + 0.5 * dt * k2q,
        br,
        btheta,
        eta,
        nu,
        lorentz_feedback,
    )
    k4o, k4q = rhs(
        grid,
        omega + dt * k3o,
        q + dt * k3q,
        br,
        btheta,
        eta,
        nu,
        lorentz_feedback,
    )
    omega_next = omega + (dt / 6.0) * (k1o + 2.0 * k2o + 2.0 * k3o + k4o)
    q_next = q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    return omega_next, q_next


def energies(
    grid: ShellGrid,
    omega: np.ndarray,
    q: np.ndarray,
    br: np.ndarray,
    btheta: np.ndarray,
) -> dict[str, float]:
    u_phi = grid.s * omega
    b_phi = grid.s * q
    e_kin = 0.5 * RHO0 * volume_integral(grid, u_phi**2)
    e_tor = 0.5 / MU0 * volume_integral(grid, b_phi**2)
    e_pol = 0.5 / MU0 * volume_integral(grid, br**2 + btheta**2)
    return {
        "kinetic": e_kin,
        "magnetic_toroidal": e_tor,
        "magnetic_poloidal": e_pol,
        "magnetic_total": e_pol + e_tor,
        "total": e_kin + e_pol + e_tor,
    }


def divergence_poloidal_fd(
    grid: ShellGrid, br: np.ndarray, btheta: np.ndarray
) -> np.ndarray:
    """Finite-difference divergence diagnostic for the represented B_p."""

    radial = d_dr(grid, grid.radius2d**2 * br, "neumann", "neumann") / grid.radius2d**2
    angular = d_dtheta(grid, np.sin(grid.theta2d) * btheta, "neumann", "neumann") / (
        grid.radius2d * np.sin(grid.theta2d)
    )
    return radial + angular


def dissipation_rates(
    grid: ShellGrid,
    omega: np.ndarray,
    q: np.ndarray,
    eta: float,
    nu: float,
) -> dict[str, float]:
    b_phi = grid.s * q
    _db, jr, jtheta = negative_curl_curl_toroidal(grid, b_phi)
    d_ohm = -eta / MU0 * volume_integral(grid, b_phi * _db)

    if nu == 0.0:
        d_visc = 0.0
    else:
        _op, omega_r, omega_theta = negative_weighted_laplacian_omega(grid, omega)
        d_visc = -RHO0 * nu * volume_integral(grid, grid.s**2 * omega * _op)
    return {"ohmic": d_ohm, "viscous": d_visc}


def diagnostics_row(
    grid: ShellGrid,
    case: dict[str, Any],
    time_value: float,
    omega: np.ndarray,
    q: np.ndarray,
    psi: np.ndarray,
    br: np.ndarray,
    btheta: np.ndarray,
    positive_zone: np.ndarray,
    eta: float,
    nu: float,
    lorentz_feedback: bool,
    cumulative_ohmic: float,
    cumulative_viscous: float,
    initial_total_energy: float,
    cumulative_driving: float = 0.0,
) -> dict[str, Any]:
    energy = energies(grid, omega, q, br, btheta)
    omega_t, q_t = rhs(grid, omega, q, br, btheta, eta, nu, lorentz_feedback)
    b_phi = grid.s * q
    u_phi = grid.s * omega
    b_phi_t = grid.s * q_t
    u_phi_t = grid.s * omega_t

    d_energy_dt = volume_integral(grid, u_phi * u_phi_t + b_phi * b_phi_t) / MU0
    diss = dissipation_rates(grid, omega, q, eta, nu)
    budget_rhs = -(diss["ohmic"] + diss["viscous"])
    denom = max(abs(d_energy_dt) + abs(budget_rhs), 1.0e-30)
    residual = abs(d_energy_dt - budget_rhs) / denom

    winding = winding_operator(grid, br, btheta, omega)
    magnetic_source_rate = volume_integral(grid, b_phi * grid.s * winding)
    if lorentz_feedback:
        omega_t_lorentz = (
            -winding_adjoint(grid, br, btheta, grid.mass_s2 * q) / grid.mass_s2
        )
    else:
        omega_t_lorentz = np.zeros_like(omega)
    lorentz_work_rate = volume_integral(grid, u_phi * grid.s * omega_t_lorentz)
    driving_power = 0.0 if lorentz_feedback else magnetic_source_rate
    budget_rhs += driving_power
    denom = max(abs(d_energy_dt) + abs(budget_rhs), 1.0e-30)
    residual = abs(d_energy_dt - budget_rhs) / denom
    torque_density = grid.s**2 * omega_t_lorentz
    north = grid.theta2d < 0.5 * math.pi
    positive = positive_zone

    div_fd = divergence_poloidal_fd(grid, br, btheta)
    div_fd_rms = volume_rms(grid, div_fd)
    b_rms = volume_rms(grid, np.sqrt(br**2 + btheta**2 + b_phi**2))
    div_fd_relative = div_fd_rms * grid.shell_length / max(b_rms, 1.0e-300)
    bs = br * np.sin(grid.theta2d) + btheta * np.cos(grid.theta2d)
    stress = bs * b_phi / MU0

    cumulative_residual = (
        energy["total"]
        - initial_total_energy
        + cumulative_ohmic
        + cumulative_viscous
        - cumulative_driving
    ) / max(initial_total_energy, 1.0e-300)
    return {
        "case": str(case["name"]),
        "suite": str(case.get("suite", "single")),
        "time": time_value,
        "ntheta": grid.ntheta,
        "nr": grid.nr,
        "r_inner": grid.r_inner,
        "r_outer": grid.r_outer,
        "lorentz_feedback": int(lorentz_feedback),
        "field_strength": float(case["field_strength"]),
        "eta": eta,
        "nu": nu,
        "E_kin": energy["kinetic"],
        "E_pol": energy["magnetic_poloidal"],
        "E_tor": energy["magnetic_toroidal"],
        "E_mag": energy["magnetic_total"],
        "E_total": energy["total"],
        "dE_dt": d_energy_dt,
        "D_ohmic": diss["ohmic"],
        "D_viscous": diss["viscous"],
        "cum_ohmic": cumulative_ohmic,
        "cum_viscous": cumulative_viscous,
        "prescribed_flow_power": driving_power,
        "cum_prescribed_flow_work": cumulative_driving,
        "energy_budget_residual": residual,
        "cumulative_energy_residual": cumulative_residual,
        "lorentz_work_rate": lorentz_work_rate,
        "magnetic_winding_power": magnetic_source_rate,
        "shear_amplitude": weighted_shear_amplitude(grid, omega),
        "delta_omega_peak_to_peak": float(np.max(omega) - np.min(omega)),
        "max_abs_B_phi": float(np.max(np.abs(b_phi))),
        "rms_B_phi": volume_rms(grid, b_phi),
        "maxwell_stress_rms": volume_rms(grid, stress),
        "maxwell_torque_abs": volume_integral(grid, np.abs(torque_density)),
        "maxwell_stress_abs": volume_integral(grid, np.abs(stress)),
        "maxwell_torque_total": volume_integral(grid, torque_density),
        "maxwell_torque_north": float(
            np.sum(grid.weights[north] * torque_density[north])
        ),
        "maxwell_torque_positive_zone": float(
            np.sum(grid.weights[positive] * torque_density[positive])
        ),
        "angular_momentum_total": volume_integral(grid, grid.s**2 * omega),
        "angular_momentum_north": float(
            np.sum(grid.weights[north] * grid.s[north] ** 2 * omega[north])
        ),
        "angular_momentum_south": float(
            np.sum(grid.weights[~north] * grid.s[~north] ** 2 * omega[~north])
        ),
        "angular_momentum_positive_zone": float(
            np.sum(grid.weights[positive] * grid.s[positive] ** 2 * omega[positive])
        ),
        "angular_momentum_negative_zone": float(
            np.sum(grid.weights[~positive] * grid.s[~positive] ** 2 * omega[~positive])
        ),
        "div_B_construction_zero": 0.0,
        "div_B_rms": 0.0,
        "div_u_rms": 0.0,
        "div_B_fd_rms": div_fd_rms,
        "div_B_fd_relative": div_fd_relative,
        "min_omega": float(np.min(omega)),
        "max_omega": float(np.max(omega)),
        "min_B_phi": float(np.min(b_phi)),
        "max_B_phi": float(np.max(b_phi)),
    }


def snapshot_fields(
    grid: ShellGrid,
    omega: np.ndarray,
    q: np.ndarray,
    psi: np.ndarray,
    br: np.ndarray,
    btheta: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "theta": grid.theta,
        "radius": grid.radius,
        "s": grid.s,
        "z": grid.z,
        "omega": omega,
        "q": q,
        "psi": psi,
        "B_r": br,
        "B_theta": btheta,
        "B_phi": grid.s * q,
        "u_phi": grid.s * omega,
    }


def save_snapshots(path: str | Path, snapshots: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        for index, snap in enumerate(snapshots):
            group = handle.create_group(f"snap_{index:03d}")
            group.attrs["time"] = float(snap["time"])
            for key, value in snap["fields"].items():
                group.create_dataset(key, data=np.asarray(value), compression="gzip")


def run_case(
    config: dict[str, Any], output_dir: str | Path | None = None
) -> dict[str, Any]:
    grid = make_grid(config)
    field_strength = float(config.get("field_strength", 0.07))
    eta = float(config.get("eta", 2.0e-5))
    nu = float(config.get("nu", 0.0))
    stop_time = float(config.get("stop_time", 60.0))
    timestep = float(config.get("timestep", 0.005))
    output_dt = float(config.get("output_dt", 0.1))
    snapshot_count = int(config.get("snapshot_count", 13))
    lorentz_feedback = bool(config.get("lorentz_feedback", True))
    if (
        not np.all(np.isfinite([timestep, output_dt, stop_time, eta, nu]))
        or timestep <= 0
        or output_dt <= 0
        or stop_time < 0
        or eta < 0
        or nu < 0
    ):
        raise ValueError(
            "require positive timestep/output_dt and nonnegative stop_time/eta/nu"
        )
    case_name = str(config.get("name", f"B{field_strength:g}_eta{eta:g}"))

    poloidal = poloidal_field(grid, field_strength)
    psi = np.asarray(poloidal["psi"], dtype=float)
    br = np.asarray(poloidal["br"], dtype=float)
    btheta = np.asarray(poloidal["btheta"], dtype=float)
    omega = initial_omega(grid, config)
    q = np.zeros_like(omega)
    positive_zone = omega > 0.0

    case = dict(config)
    case["name"] = case_name
    case["field_strength"] = field_strength

    initial_energy = energies(grid, omega, q, br, btheta)
    initial_total = initial_energy["total"]
    output_times = np.r_[np.arange(0.0, stop_time - 1e-12, output_dt), stop_time]
    output_times[0] = 0.0
    snapshot_times = np.linspace(0.0, stop_time, snapshot_count)
    next_output = 0
    next_snapshot = 0
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    cumulative_ohmic = 0.0
    cumulative_viscous = 0.0
    cumulative_driving = 0.0
    driving_prev = 0.0
    diss_prev = dissipation_rates(grid, omega, q, eta, nu)
    t = 0.0
    start = time.perf_counter()

    def record_outputs() -> None:
        nonlocal next_output, next_snapshot
        while (
            next_output < len(output_times) and t >= output_times[next_output] - 1.0e-12
        ):
            rows.append(
                diagnostics_row(
                    grid,
                    case,
                    float(output_times[next_output]),
                    omega,
                    q,
                    psi,
                    br,
                    btheta,
                    positive_zone,
                    eta,
                    nu,
                    lorentz_feedback,
                    cumulative_ohmic,
                    cumulative_viscous,
                    initial_total,
                    cumulative_driving,
                )
            )
            next_output += 1
        while (
            next_snapshot < len(snapshot_times)
            and t >= snapshot_times[next_snapshot] - 1.0e-12
        ):
            snapshots.append(
                {
                    "time": float(snapshot_times[next_snapshot]),
                    "fields": snapshot_fields(grid, omega, q, psi, br, btheta),
                }
            )
            next_snapshot += 1

    record_outputs()
    while t < stop_time - 1.0e-12:
        dt = min(timestep, stop_time - t)
        if next_output < len(output_times):
            dt = min(dt, output_times[next_output] - t)
        if next_snapshot < len(snapshot_times):
            dt = min(dt, snapshot_times[next_snapshot] - t)
        omega, q = rk4_step(grid, omega, q, br, btheta, dt, eta, nu, lorentz_feedback)
        if not (np.all(np.isfinite(omega)) and np.all(np.isfinite(q))):
            raise FloatingPointError(
                f"non-finite field encountered in {case_name} at t={t:g}"
            )
        t += dt
        diss_now = dissipation_rates(grid, omega, q, eta, nu)
        cumulative_ohmic += 0.5 * (diss_prev["ohmic"] + diss_now["ohmic"]) * dt
        cumulative_viscous += 0.5 * (diss_prev["viscous"] + diss_now["viscous"]) * dt
        diss_prev = diss_now
        if not lorentz_feedback:
            driving_now = volume_integral(
                grid, grid.s**2 * q * winding_operator(grid, br, btheta, omega)
            )
            cumulative_driving += 0.5 * (driving_prev + driving_now) * dt
            driving_prev = driving_now
        record_outputs()

    runtime = time.perf_counter() - start
    summary = summarize_rows(rows)
    summary.update(
        {
            "case": case_name,
            "suite": str(config.get("suite", "single")),
            "ntheta": grid.ntheta,
            "nr": grid.nr,
            "field_strength": field_strength,
            "eta": eta,
            "nu": nu,
            "lorentz_feedback": int(lorentz_feedback),
            "runtime_seconds": runtime,
            "alfven_time_shell": grid.shell_length / max(field_strength, 1.0e-300),
            "reynolds_number": "unbounded_nu0" if nu == 0.0 else grid.shell_length / nu,
            "magnetic_reynolds_number": "unbounded_eta0"
            if eta == 0.0
            else grid.shell_length / eta,
            "magnetic_prandtl_number": "undefined" if eta == 0.0 else nu / eta,
            "lundquist_number": "unbounded_eta0"
            if eta == 0.0
            else field_strength * grid.shell_length / eta,
        }
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "diagnostics.csv", rows)
        save_snapshots(out / "snapshots.h5", snapshots)
        params = dict(config)
        params.update(
            {
                "actual_runtime_seconds": runtime,
                "backend": "axisymmetric_energy_adjoint_finite_difference",
                "poloidal_field": "anchored shell-confined dipolar flux function",
                "notes": "B_phi is evolved resistively; the poloidal field is fixed and solenoidal by construction.",
            }
        )
        dump_yaml(out / "run_parameters.yaml", params)
    return {"rows": rows, "summary": summary, "snapshots": snapshots}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize empty diagnostics")
    e0 = float(rows[0]["E_kin"])
    initial_shear = float(rows[0]["shear_amplitude"])
    braking_time: float | None = None
    for row in rows:
        if e0 > 0 and float(row["E_kin"]) <= 0.5 * e0:
            braking_time = float(row["time"])
            break
    peak_b = max(rows, key=lambda row: float(row["max_abs_B_phi"]))
    peak_rms_b = max(rows, key=lambda row: float(row["rms_B_phi"]))
    peak_tor = max(rows, key=lambda row: float(row["E_tor"]))
    return {
        "initial_E_kin": e0,
        "initial_E_pol": float(rows[0]["E_pol"]),
        "initial_E_tor": float(rows[0]["E_tor"]),
        "initial_shear_amplitude": initial_shear,
        "braking_time_half_Ekin": braking_time
        if braking_time is not None
        else "not_reached",
        "peak_abs_B_phi": float(peak_b["max_abs_B_phi"]),
        "time_peak_abs_B_phi": float(peak_b["time"]),
        "peak_rms_B_phi": float(peak_rms_b["rms_B_phi"]),
        "time_peak_rms_B_phi": float(peak_rms_b["time"]),
        "peak_E_tor": float(peak_tor["E_tor"]),
        "time_peak_E_tor": float(peak_tor["time"]),
        "final_E_kin": float(rows[-1]["E_kin"]),
        "final_E_tor": float(rows[-1]["E_tor"]),
        "final_E_total": float(rows[-1]["E_total"]),
        "final_shear_amplitude": float(rows[-1]["shear_amplitude"]),
        "final_shear_fraction": float(rows[-1]["shear_amplitude"])
        / max(initial_shear, 1.0e-300),
        "max_energy_budget_residual": max(
            abs(float(row["energy_budget_residual"])) for row in rows
        ),
        "max_cumulative_energy_residual": max(
            abs(float(row["cumulative_energy_residual"])) for row in rows
        ),
        "final_cumulative_energy_residual": float(
            rows[-1]["cumulative_energy_residual"]
        ),
        "max_div_B_rms": max(float(row["div_B_rms"]) for row in rows),
        "max_div_u_rms": max(float(row["div_u_rms"]) for row in rows),
        "max_div_B_fd_relative": max(float(row["div_B_fd_relative"]) for row in rows),
        "max_abs_torque_transport": max(
            abs(float(row["maxwell_torque_abs"])) for row in rows
        ),
        "max_abs_positive_zone_torque": max(
            abs(float(row["maxwell_torque_positive_zone"])) for row in rows
        ),
        "max_abs_north_torque": max(
            abs(float(row["maxwell_torque_north"])) for row in rows
        ),
        "max_abs_total_torque": max(
            abs(float(row["maxwell_torque_total"])) for row in rows
        ),
    }
