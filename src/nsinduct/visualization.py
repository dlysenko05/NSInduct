"""Cartesian reconstruction of the reduced axisymmetric solution.

Only spatial interpolation is used. Frames retain their saved physical times.
The prescribed poloidal field is evaluated analytically in its actual geometry.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator


def spherical_to_cartesian(br, btheta, bphi, theta, phi):
    """Transform components in the orthonormal (r, colatitude, azimuth) basis."""
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    return np.stack(
        np.broadcast_arrays(
            br * st * cp + btheta * ct * cp - bphi * sp,
            br * st * sp + btheta * ct * sp + bphi * cp,
            br * ct - btheta * st,
        ),
        axis=-1,
    )


class CartesianField:
    """Interpolate regular q/Omega; reconstruct Bphi=sq and uphi=sOmega."""

    def __init__(self, snapshot, r_inner=0.3, r_outer=1.0):
        self.time = float(snapshot["time"])
        self.fields = snapshot["fields"]
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)
        f = self.fields
        t = np.asarray(f["theta"])
        r = np.asarray(f["radius"])
        self.theta = np.r_[0.0, t, np.pi]
        self.radius = np.r_[r_inner, r, r_outer]
        self.interpolators = {}
        for key in ["q", "omega"]:
            values = np.asarray(f[key])
            extended = np.pad(values, ((1, 1), (1, 1)), mode="edge")
            # Bounded even extension gives zero polar slope without creating extrema.
            extended[0, 1:-1] = values[0]
            extended[-1, 1:-1] = values[-1]
            if key == "q":
                extended[:, [0, -1]] = 0
            else:
                extended[:, 0] = extended[:, 1]
                extended[:, -1] = extended[:, -2]
            self.interpolators[key] = RegularGridInterpolator(
                (self.theta, self.radius), extended, bounds_error=True
            )
        rr, tt = np.meshgrid(r, t)
        h = ((rr - r_inner) * (r_outer - rr)) ** 2
        raw = 2 * h * np.cos(tt) / rr**2
        self.poloidal_scale = float(np.sum(raw * np.asarray(f["B_r"])) / np.sum(raw**2))
        hp = 2 * (rr - r_inner) * (r_outer - rr) * (r_inner + r_outer - 2 * rr)
        expected_bt = -self.poloidal_scale * hp * np.sin(tt) / rr
        if not (
            np.allclose(f["B_r"], self.poloidal_scale * raw, rtol=1e-10, atol=1e-14)
            and np.allclose(f["B_theta"], expected_bt, rtol=1e-10, atol=1e-14)
        ):
            raise ValueError(
                "Cartesian reconstruction requires the confined quartic poloidal field"
            )

    def sample(self, points):
        """Return physical vectors/scalars; outside-shell values are NaN, mask=False."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        r = np.linalg.norm(points, axis=1)
        tol = 32 * np.finfo(float).eps
        mask = (r >= self.r_inner - tol) & (r <= self.r_outer + tol) & np.isfinite(r)
        result = {key: np.full(len(r), np.nan) for key in ["q", "omega", "Bphi"]}
        result.update(
            B=np.full((len(r), 3), np.nan), u=np.full((len(r), 3), np.nan), mask=mask
        )
        if not np.any(mask):
            return result
        rm = np.clip(r[mask], self.r_inner, self.r_outer)
        theta = np.arccos(np.clip(points[mask, 2] / r[mask], -1.0, 1.0))
        phi = np.arctan2(points[mask, 1], points[mask, 0])
        query = np.c_[theta, rm]
        q = self.interpolators["q"](query)
        omega = self.interpolators["omega"](query)
        ss = rm * np.sin(theta)
        bphi = ss * q
        uphi = ss * omega
        h = ((rm - self.r_inner) * (self.r_outer - rm)) ** 2
        hp = (
            2
            * (rm - self.r_inner)
            * (self.r_outer - rm)
            * (self.r_inner + self.r_outer - 2 * rm)
        )
        br = self.poloidal_scale * 2 * h * np.cos(theta) / rm**2
        bt = -self.poloidal_scale * hp * np.sin(theta) / rm
        result["q"][mask] = q
        result["omega"][mask] = omega
        result["Bphi"][mask] = bphi
        result["B"][mask] = spherical_to_cartesian(br, bt, bphi, theta, phi)
        result["u"][mask] = spherical_to_cartesian(
            np.zeros_like(uphi), np.zeros_like(uphi), uphi, theta, phi
        )
        return result


def fixed_color_limits(snapshots):
    """Symmetric signed ranges shared by every physical-time frame."""
    limits = {}
    for name, key in [("Bphi", "B_phi"), ("omega", "omega")]:
        value = max(float(np.max(abs(s["fields"][key]))) for s in snapshots)
        limits[name] = (-value, value)
    return limits


def integrate_streamline(field, seed, max_length=12.0, step=0.01, direction="both"):
    """Integrate dX/dl=±B/|B| in Cartesian space, stopping at shell boundaries."""
    if direction not in {"both", "forward", "backward"}:
        raise ValueError("Invalid direction")
    seed = np.asarray(seed, dtype=float).reshape(3)
    if not field.sample(seed)["mask"][0]:
        raise ValueError("Streamline seed is outside shell")

    def part(sign):
        def fun(_, x):
            b = field.sample(x)["B"][0]
            bn = np.linalg.norm(b)
            return sign * b / bn if np.isfinite(bn) and bn > 1e-12 else np.zeros(3)

        def inner(_, x):
            return np.linalg.norm(x) - field.r_inner - 1e-7

        def outer(_, x):
            return field.r_outer - 1e-7 - np.linalg.norm(x)

        inner.terminal = True
        outer.terminal = True
        sol = solve_ivp(
            fun,
            (0, max_length),
            seed,
            max_step=step,
            rtol=1e-7,
            atol=1e-9,
            events=[inner, outer],
        )
        return sol.y.T

    if direction == "forward":
        return part(1)
    if direction == "backward":
        return part(-1)
    backward = part(-1)
    forward = part(1)
    return np.vstack([backward[::-1], forward[1:]])
