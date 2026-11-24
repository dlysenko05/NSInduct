"""Reproduce numerical studies or clean-kernel notebooks in an external directory."""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nsinduct import shell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Use reduced demonstration grids"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="External directory for generated results",
    )
    parser.add_argument(
        "--notebooks-only",
        action="store_true",
        help="Execute the three notebooks instead of the parameter studies",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.expanduser().resolve()
    if out.is_relative_to(root):
        parser.error("Choose an output directory outside the source repository")
    out.mkdir(parents=True, exist_ok=True)
    if args.notebooks_only:
        execute_notebooks(root, out)
        return
    base = dict(
        ntheta=16 if args.quick else 48,
        nr=24 if args.quick else 72,
        field_strength=0.07,
        eta=2e-5,
        timestep=0.01 if args.quick else 0.005,
        stop_time=20 if args.quick else 30,
        output_dt=0.1,
        snapshot_count=2,
    )
    data = {
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "experiments": [],
    }

    def run(label, **params):
        config = base | params | {"name": label}
        res = shell.run_case(config)
        data["experiments"].append({"config": config, "summary": res["summary"]})
        shell.write_csv(out / (label + ".csv"), res["rows"])
        (
            out / ("validation_quick.json" if args.quick else "validation.json")
        ).write_text(json.dumps(data, indent=2))
        print(
            label,
            res["summary"]["braking_time_half_Ekin"],
            res["summary"]["peak_E_tor"],
            flush=True,
        )
        return res

    for b in [0.035, 0.07, 0.14]:
        run("field_" + str(b), field_strength=b)
    for eta in [0, 2e-5, 1e-4, 5e-4]:
        run("eta_" + str(eta), eta=eta)
    run("control", lorentz_feedback=False)
    spatial = []
    for n in [12, 16, 24] if args.quick else [24, 32, 48, 64]:
        res = run("grid_" + str(n), ntheta=n, nr=3 * n // 2, stop_time=10)
        spatial.append((n, res["snapshots"][-1]["fields"]))
    fine = spatial[-1][1]
    comparisons = []
    for n, f in spatial[:-1]:
        theta, r = np.meshgrid(f["theta"], f["radius"], indexing="ij")
        pts = np.c_[theta.ravel(), r.ravel()]
        g = shell.make_grid(dict(ntheta=n, nr=3 * n // 2))
        row = {"ntheta": n}
        for key in ["omega", "q"]:
            ref = RegularGridInterpolator(
                (fine["theta"], fine["radius"]),
                fine[key],
                bounds_error=False,
                fill_value=None,
            )(pts).reshape(g.s.shape)
            row[key + "_weighted_relative_L2"] = shell.volume_rms(
                g, g.s * (f[key] - ref)
            ) / shell.volume_rms(g, g.s * ref)
            row[key + "_max_error"] = float(np.max(abs(f[key] - ref)))
        comparisons.append(row)
    data["spatial_comparisons_to_finest"] = comparisons
    temporal = []
    for dt in [0.04, 0.02, 0.01, 0.005]:
        res = run("dt_" + str(dt), ntheta=16, nr=24, timestep=dt, stop_time=10)
        temporal.append((dt, res["snapshots"][-1]["fields"]))
    g = shell.make_grid(dict(ntheta=16, nr=24))
    tf = temporal[-1][1]
    data["temporal_comparisons_to_finest"] = [
        {
            "dt": dt,
            **{
                k + "_weighted_relative_L2": shell.volume_rms(g, g.s * (f[k] - tf[k]))
                / shell.volume_rms(g, g.s * tf[k])
                for k in ["omega", "q"]
            },
        }
        for dt, f in temporal[:-1]
    ]
    (out / ("validation_quick.json" if args.quick else "validation.json")).write_text(
        json.dumps(data, indent=2)
    )


def execute_notebooks(root, out):
    import os
    import tempfile

    import nbformat
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    with tempfile.TemporaryDirectory(prefix="nsinduct-kernel-") as temporary:
        spec = Path(temporary) / "nsinduct-local"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-I",
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "NSInduct",
                    "language": "python",
                }
            )
        )
        manager = KernelSpecManager(kernel_dirs=[temporary])
        for path in sorted((root / "notebooks").glob("0*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=600,
                kernel_name="nsinduct-local",
                resources={"metadata": {"path": str(out)}},
                km=AsyncKernelManager(
                    kernel_name="nsinduct-local", kernel_spec_manager=manager
                ),
            )
            client.execute(cleanup_kc=True)
            nbformat.write(notebook, out / path.name)
            print("Executed", path.name, flush=True)


if __name__ == "__main__":
    main()
