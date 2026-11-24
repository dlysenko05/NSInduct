"""Plot four meridional snapshots of the finite-difference braking run."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nsinduct.shell import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_case(
        dict(
            r_inner=0.3,
            r_outer=1.0,
            ntheta=48,
            nr=72,
            delta_omega=1.0,
            radial_contrast=0.18,
            field_strength=0.07,
            eta=2e-5,
            nu=0.0,
            lorentz_feedback=True,
            stop_time=80.0,
            timestep=0.005,
            output_dt=0.1,
            snapshot_count=13,
        )
    )
    snapshots = result["snapshots"]
    selected = [snapshots[i] for i in (0, 1, 2, 12)]
    limit = max(np.max(np.abs(snap["fields"]["B_phi"])) for snap in selected)
    theta_edges = np.linspace(0, np.pi, 49)[:, None]
    radius_edges = np.linspace(0.3, 1.0, 73)[None, :]
    s_edges = radius_edges * np.sin(theta_edges)
    z_edges = radius_edges * np.cos(theta_edges)

    fig, axes = plt.subplots(
        1, 4, figsize=(9, 3.3), sharex=True, sharey=True, layout="constrained"
    )
    for ax, snap in zip(axes, selected):
        f = snap["fields"]
        color = ax.pcolormesh(
            s_edges, z_edges, f["B_phi"], cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        for field, line_color, count in [("psi", "0.25", 8), ("omega", "white", 6)]:
            values = f[field]
            levels = np.linspace(values.min(), values.max(), count)[1:-1]
            ax.contour(f["s"], f["z"], values, levels, colors=line_color, linewidths=0.55)
        for radius in (0.3, 1.0):
            ax.plot(radius * np.sin(theta_edges), radius * np.cos(theta_edges), color="0.2")
        ax.set(title=rf"$t={snap['time']:.1f}$", xlabel=r"$s$", aspect="equal")
    axes[0].set_ylabel(r"$z$")
    fig.colorbar(color, ax=axes, shrink=0.8, label=r"$B_\phi$")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    summary = result["summary"]
    print("Sampled kinetic-energy half-time:", summary["braking_time_half_Ekin"])


if __name__ == "__main__":
    main()
