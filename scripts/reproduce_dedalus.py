"""Bounded serial reproduction of the toroidal-vector Dedalus PDE studies."""

import argparse
import json
from pathlib import Path

from nsinduct.dedalus_backend import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite", choices=["quick", "braking"], default="quick")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    if out.is_relative_to(root):
        raise ValueError("Choose an external output directory")
    out.mkdir(parents=True, exist_ok=True)
    base = dict(
        ntheta=16,
        nr=24,
        timestep=0.01,
        stop_time=2.0,
        output_dt=0.1,
        snapshot_count=5,
        eta=2e-5,
        nu=0.0,
        field_strength=0.07,
        max_wall_seconds=120,
        max_memory_mb=2048,
    )
    cases = [
        ("feedback", {}),
        ("control", dict(lorentz_feedback=False)),
        ("ideal", dict(eta=0.0)),
    ]
    if args.suite == "braking":
        base.update(ntheta=32, nr=48, snapshot_count=21, max_wall_seconds=300)
        cases = [
            ("field_weak", dict(field_strength=0.035)),
            ("field_intermediate", {}),
            ("field_strong", dict(field_strength=0.14)),
            ("control", dict(lorentz_feedback=False)),
            ("eta_ideal", dict(eta=0.0)),
            ("eta_intermediate", dict(eta=1e-4)),
            ("eta_strong", dict(eta=5e-4)),
        ]
    summaries = []
    for name, extra in cases:
        config = base | extra | {"name": name}
        if args.suite == "braking":
            strength = config["field_strength"]
            config.update(stop_time=0.98 / strength, timestep=0.0007 / strength)
            if config["eta"] == 0:
                config["stop_time"] = min(config["stop_time"], 12.0)
        result = run_case(config, out / name)
        summaries.append(
            dict(name=name, summary=result["summary"], metadata=result["metadata"])
        )
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(name, result["summary"], flush=True)


if __name__ == "__main__":
    main()
