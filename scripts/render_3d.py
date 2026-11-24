"""Render axisymmetric torsional solutions; every physical frame is a saved state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from nsinduct.visualization import (
    CartesianField,
    fixed_color_limits,
    integrate_streamline,
)

from nsinduct import shell


def output_path(value):
    path = Path(value).expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if path.is_relative_to(repo):
        raise argparse.ArgumentTypeError(
            "Choose an output directory outside the source repository"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_data(out, backend="finite_difference"):
    config = dict(
        name="visualization",
        ntheta=48,
        nr=72,
        field_strength=0.07,
        eta=2e-5,
        nu=0.0,
        stop_time=16.0,
        timestep=0.005,
        output_dt=0.1,
        snapshot_count=33,
    )
    solver_module = shell
    if backend == "dedalus":
        from nsinduct import dedalus_backend as solver_module

        config.update(ntheta=32, nr=48, timestep=0.005, max_wall_seconds=300)
    source_hash = hashlib.sha256(Path(solver_module.__file__).read_bytes()).hexdigest()
    cache = out / "simulation"
    if (cache / "snapshots.h5").exists() and (cache / "provenance.json").exists():
        old = json.loads((cache / "provenance.json").read_text())
        if old["config"] != config or old["solver_sha256"] != source_hash:
            raise RuntimeError(
                "Existing simulation has different source/config; use a fresh output directory"
            )
        snapshots = []
        with h5py.File(cache / "snapshots.h5") as store:
            for group in store.values():
                snapshots.append(
                    dict(
                        time=float(group.attrs["time"]),
                        fields={k: group[k][...] for k in group},
                    )
                )
    else:
        result = solver_module.run_case(config, cache)
        snapshots = result["snapshots"]
        (cache / "provenance.json").write_text(
            json.dumps(dict(config=config, solver_sha256=source_hash), indent=2)
        )
    np.savez_compressed(
        cache / "fields.npz",
        time=np.array([s["time"] for s in snapshots]),
        theta=snapshots[0]["fields"]["theta"],
        radius=snapshots[0]["fields"]["radius"],
        q=np.stack([s["fields"]["q"] for s in snapshots]),
        omega=np.stack([s["fields"]["omega"] for s in snapshots]),
        B_r=snapshots[0]["fields"]["B_r"],
        B_theta=snapshots[0]["fields"]["B_theta"],
    )
    return snapshots, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=[
            "all",
            "magnetic",
            "rotation",
            "toroidal",
            "animations",
            "magnetic-evolution",
            "rotation-evolution",
            "camera",
        ],
        default="all",
    )
    parser.add_argument("--output-dir", required=True, type=output_path)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render principal morphology only for review",
    )
    parser.add_argument(
        "--backend",
        choices=["finite_difference", "dedalus"],
        default="finite_difference",
    )
    args = parser.parse_args()
    out = args.output_dir
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import pyvista as pv
    from nsinduct.render_layout import compose, scene_size
    from PIL import Image

    pv.global_theme.font.family = "arial"
    pv.global_theme.font.color = "#172637"
    snaps, config = prepare_data(out, args.backend)
    limits = fixed_color_limits(snaps)
    morphology = snaps[4]  # t=2, a resolved early winding state.
    later = snaps[24]  # t=12, after the first energy transfer maximum.
    records = {
        "config": config,
        "times": [s["time"] for s in snaps],
        "limits": limits,
        "camera": {
            "position": [3.4, -5, 2.7],
            "focal_point": [0, 0, 0],
            "up": [0, 0, 1],
        },
        "seed_azimuths": [0.1, float(np.pi + 0.1)],
        "seed_radius_theta": [[0.47, 0.9], [0.65, 0.9], [0.81, 1.15]],
        "streamline_length_each_direction": 4.5,
        "presentation_parallel_scales": {"oblique": 1.16, "orthogonal": 1.10},
        "morphology_time": morphology["time"],
        "braked_time": later["time"],
        "representation": "3D reconstruction of an axisymmetric shell solution",
        "backend": args.backend,
        "render_versions": {
            "pyvista": pv.__version__,
            "vtk": pv.vtk_version_info.__str__(),
        },
    }

    def plotter(shape=None, size=None):
        size = scene_size() if size is None else size
        p = pv.Plotter(
            off_screen=True, window_size=size, shape=shape or (1, 1), border=False
        )
        p.set_background("#fafbfc")
        p.enable_anti_aliasing("ssaa")
        return p

    def orient(p, azimuth=None):
        angle = np.deg2rad(0.0 if azimuth is None else azimuth)
        x, y = 3.4, -5.0
        position = (
            x * np.cos(angle) - y * np.sin(angle),
            x * np.sin(angle) + y * np.cos(angle),
            2.7,
        )
        p.camera_position = [position, (0, 0, 0), (0, 0, 1)]
        p.enable_parallel_projection()
        p.camera.parallel_scale = 1.16
        p.add_axes(line_width=3, labels_off=False, xlabel="x", ylabel="y", zlabel="z")
        p.add_mesh(pv.Line((0, 0, -1.06), (0, 0, 1.10)), color="#465466", line_width=2)

    def scaffold(p, opacity=0.10):
        back = pv.Sphere(radius=1.0, theta_resolution=100, phi_resolution=80).clip(
            normal=(0, -1, 0), invert=False
        )
        p.add_mesh(back, color="#bec9d1", opacity=opacity, smooth_shading=True)
        p.add_mesh(
            pv.Sphere(radius=0.3, theta_resolution=64, phi_resolution=48),
            color="#c2ccd4",
            opacity=0.48,
            smooth_shading=True,
        )
        for radius in [0.3, 1.0]:
            angle = np.linspace(0, 2 * np.pi, 240)
            points = np.c_[
                radius * np.sin(angle), np.zeros(len(angle)), radius * np.cos(angle)
            ]
            p.add_mesh(pv.lines_from_points(points), color="#8393a0", line_width=1)

    def cut_mesh(field, phi=0.0):
        theta = np.linspace(0, np.pi, 150)
        radius = np.linspace(0.3, 1.0, 100)
        rr, tt = np.meshgrid(radius, theta, indexing="ij")
        points = np.stack(
            [
                rr * np.sin(tt) * np.cos(phi),
                rr * np.sin(tt) * np.sin(phi),
                rr * np.cos(tt),
            ],
            axis=-1,
        )
        mesh = pv.StructuredGrid(points[..., 0], points[..., 1], points[..., 2])
        samples = field.sample(mesh.points)
        mesh["Bphi"] = samples["Bphi"]
        mesh["omega"] = samples["omega"]
        return mesh

    def scalar_scene(snapshot, key, p=None, azimuth=None):
        p = plotter() if p is None else p
        field = CartesianField(snapshot)
        scaffold(p, opacity=0.08)
        for phi in [0.0, np.pi]:
            p.add_mesh(
                cut_mesh(field, phi),
                scalars=key,
                cmap="RdBu_r",
                clim=limits[key],
                show_scalar_bar=False,
                lighting=False,
            )
        orient(p, azimuth)
        return p

    line_cache = []

    def morphology_scene(p=None, azimuth=None):
        p = plotter() if p is None else p
        field = CartesianField(morphology)
        scaffold(p, opacity=0.11)
        lines = line_cache
        for phi in [] if lines else [0.1, np.pi + 0.1]:
            for radius, theta in [(0.47, 0.9), (0.65, 0.9), (0.81, 1.15)]:
                seed = np.array(
                    [
                        radius * np.sin(theta) * np.cos(phi),
                        radius * np.sin(theta) * np.sin(phi),
                        radius * np.cos(theta),
                    ]
                )
                line = integrate_streamline(field, seed, max_length=4.5, step=0.015)
                lines.append(line)
        vmax = max(
            float(np.max(np.linalg.norm(field.sample(line)["B"], axis=1)))
            for line in lines
        )
        for line in lines:
            curve = pv.lines_from_points(line)
            strength = np.linalg.norm(field.sample(line)["B"], axis=1)
            curve["|B|"] = strength
            p.add_mesh(
                curve.tube(radius=0.007, n_sides=10),
                scalars="|B|",
                cmap="viridis",
                clim=(0, vmax),
                smooth_shading=True,
                show_scalar_bar=False,
            )
            index = int(np.argmin(line[:, 1]))
            pos = line[index]
            b = field.sample(pos)["B"][0]
            unit = b / np.linalg.norm(b)
            arrow = pv.Arrow(
                start=pos - 0.06 * unit,
                direction=unit,
                scale=0.16,
                tip_length=0.4,
                tip_radius=0.14,
                shaft_radius=0.025,
            )
            p.add_mesh(arrow, color="#ce5328", smooth_shading=True)
        orient(p, azimuth)
        return p, lines

    def composed(p, kind, snapshot=None, size=(1280, 1040), camera=False):
        footer = None
        panels = ()
        if kind == "magnetic":
            title = "Magnetic field lines"
            metadata = rf"$t = {morphology['time']:.2f}$"
            scale_label = r"$|\mathbf{B}|$"
            cmap = "viridis"
            field = CartesianField(morphology)
            vmax = max(
                float(np.max(np.linalg.norm(field.sample(line)["B"], axis=1)))
                for line in line_cache
            )
            scale = (0, vmax)
            if camera:
                footer = "Camera orbit — fixed physical state"
        elif kind == "comparison":
            title = "Differential rotation"
            metadata = r"Arrows: $0.18\mathbf{u}$"
            panels = (
                rf"Initial | $t = {snaps[0]['time']:.2f}$",
                rf"After feedback | $t = {later['time']:.2f}$",
            )
            scale_label = r"$\Omega$"
            cmap = "RdBu_r"
            scale = limits["omega"]
        else:
            key = kind
            title = (
                "Toroidal magnetic field" if key == "Bphi" else "Differential rotation"
            )
            metadata = rf"$t = {snapshot['time']:.2f}$"
            scale_label = (
                r"$B_\phi$" if key == "Bphi" else r"$\Omega$"
            )
            cmap = "RdBu_r"
            scale = limits[key]
        pixels, _ = compose(
            p.screenshot(),
            title=title,
            subtitle="",
            metadata=metadata,
            scale_label=scale_label,
            limits=scale,
            cmap=cmap,
            size=size,
            panel_labels=panels,
            footer=footer,
        )
        return pixels

    if args.preview or args.scene in ["all", "magnetic"]:
        p, lines = morphology_scene()
        try:
            Image.fromarray(composed(p, "magnetic")).save(out / "magnetic_field_3d.png")
        finally:
            p.close()
        np.savez_compressed(
            out / "morphology_streamlines.npz",
            **{f"line_{i:02d}": line for i, line in enumerate(lines)},
        )
        print("Rendered magnetic_field_3d.png", flush=True)
    if not args.preview and args.scene in ["all", "toroidal"]:
        p = scalar_scene(snaps[19], "Bphi")
        try:
            Image.fromarray(composed(p, "Bphi", snaps[19])).save(
                out / "toroidal_field_3d.png"
            )
        finally:
            p.close()
        p = scalar_scene(snaps[19], "Bphi")
        p.view_xz()
        p.camera.parallel_scale = 1.10
        try:
            Image.fromarray(composed(p, "Bphi", snaps[19])).save(
                out / "meridional_cross_section.png"
            )
        finally:
            p.close()
    if not args.preview and args.scene in ["all", "rotation"]:
        p = plotter(shape=(1, 2), size=scene_size((1600, 1040), 2))
        try:
            for index, snapshot in enumerate([snaps[0], later]):
                p.subplot(0, index)
                field = CartesianField(snapshot)
                scaffold(p, opacity=0.08)
                for phi in [0.0, np.pi]:
                    p.add_mesh(
                        cut_mesh(field, phi),
                        scalars="omega",
                        cmap="RdBu_r",
                        clim=limits["omega"],
                        show_scalar_bar=False,
                        lighting=False,
                    )
                seeds = []
                for z in [-0.55, 0, 0.55]:
                    for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        seeds.append([0.68 * np.cos(phi), 0.68 * np.sin(phi), z])
                seeds = np.array(seeds)
                u = field.sample(seeds)["u"]
                p.add_arrows(
                    seeds, 0.18 * u, mag=1, color="#192e43", show_scalar_bar=False
                )
                orient(p)
                p.camera.parallel_scale = 1.16
            Image.fromarray(composed(p, "comparison", size=(1600, 1040))).save(
                out / "differential_rotation_3d.png"
            )
        finally:
            p.close()
    if not args.preview and args.scene in [
        "all",
        "animations",
        "magnetic-evolution",
        "rotation-evolution",
        "camera",
    ]:

        def save_gif(frames, name, duration):
            palette = Image.fromarray(
                np.concatenate(
                    [frames[0], frames[len(frames) // 2], frames[-1]], axis=0
                )
            ).quantize(colors=256)
            images = [
                Image.fromarray(frame).quantize(
                    palette=palette, dither=Image.Dither.NONE
                )
                for frame in frames
            ]
            images[0].save(
                out / name,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                optimize=False,
            )

        for key, name, scene in [
            ("Bphi", "magnetic_evolution.gif", "magnetic-evolution"),
            ("omega", "differential_rotation_evolution.gif", "rotation-evolution"),
        ]:
            if args.scene not in ["all", "animations", scene]:
                continue
            frames = []
            p = plotter()
            try:
                for snapshot in snaps:
                    p.clear()
                    scalar_scene(snapshot, key, p=p)
                    frames.append(composed(p, key, snapshot))
                    print(name, snapshot["time"], flush=True)
            finally:
                p.close()
            save_gif(frames, name, 150)
        if args.scene in ["all", "animations", "camera"]:
            frames = []
            p = plotter()
            try:
                for angle in np.linspace(0, 360, 36, endpoint=False):
                    p.clear()
                    morphology_scene(p=p, azimuth=float(angle))
                    frames.append(composed(p, "magnetic", camera=True))
            finally:
                p.close()
            save_gif(frames, "camera_rotation.gif", 90)
    (out / "render_manifest.json").write_text(json.dumps(records, indent=2))
    print(out, flush=True)


if __name__ == "__main__":
    main()
