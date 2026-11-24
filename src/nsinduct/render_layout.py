"""Measured figure composition with annotation bands outside the 3D viewport."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from matplotlib import colormaps, font_manager, mathtext, rc_context
from PIL import Image, ImageDraw, ImageFont

BACKGROUND = "#fafbfc"
INK = "#172637"


def regions(size=(1280, 1040), panels=1):
    """Pixel regions scaled with the requested output resolution."""
    width, height = size
    return {
        "header": (0, 0, width, round(height * (0.21 if panels > 1 else 0.17))),
        "scene": (
            0,
            round(height * (0.22 if panels > 1 else 0.18)),
            width,
            round(height * 0.79),
        ),
        "colorbar": (0, round(height * 0.80), width, round(height * 0.92)),
        "footer": (0, round(height * 0.93), width, height),
    }


def scene_size(size=(1280, 1040), panels=1):
    box = regions(size, panels)["scene"]
    return box[2] - box[0], box[3] - box[1]


def compose(
    scene,
    *,
    title,
    subtitle,
    metadata,
    scale_label,
    limits,
    cmap,
    size=(1280, 1040),
    panel_labels=(),
    footer=None,
):
    """Return RGB pixels and measured text boxes; never draw over the scene."""
    width, height = size
    bands = regions(size, len(panel_labels) or 1)
    image = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(image)
    font_path = font_manager.findfont("DejaVu Sans")
    measurements = []
    unit = height / 1040

    def text(value, x, y, points, region, center=False, max_width=None):
        points = max(8, round(points * unit))
        available = max_width or width * 0.94
        while True:
            font = ImageFont.truetype(font_path, points)
            math_image = None
            if "$" in value:
                buffer = BytesIO()
                with rc_context({"savefig.transparent": True}):
                    mathtext.math_to_image(
                        value, buffer, prop=font_manager.FontProperties(size=points),
                        dpi=72, format="png", color=INK,
                    )
                buffer.seek(0)
                math_image = Image.open(buffer).convert("RGBA")
                box = (0, 0, math_image.width, math_image.height)
            else:
                box = draw.textbbox((0, 0), value, font=font)
            if box[2] - box[0] <= available or points <= 8:
                break
            points -= 1
        text_width, text_height = box[2] - box[0], box[3] - box[1]
        if center:
            x -= text_width / 2
        if math_image is None:
            draw.text((round(x) - box[0], round(y) - box[1]), value, font=font, fill=INK)
        else:
            image.paste(math_image, (round(x), round(y)), math_image)
        measured = (round(x), round(y), round(x) + text_width, round(y) + text_height)
        bounds = bands[region]
        if not (
            bounds[0] <= measured[0] <= measured[2] <= bounds[2]
            and bounds[1] <= measured[1] <= measured[3] <= bounds[3]
        ):
            raise ValueError(f"Text exceeds reserved {region} region: {value}")
        measurements.append({"text": value, "region": region, "bbox": measured})

    text(title, width * 0.035, height * 0.025, 36, "header")
    text(subtitle, width * 0.035, height * 0.083, 21, "header")
    text(metadata, width * 0.035, height * 0.126, 20, "header")
    for index, label in enumerate(panel_labels):
        text(
            label,
            width * (index + 0.5) / len(panel_labels),
            height * 0.18,
            23,
            "header",
            center=True,
            max_width=width / len(panel_labels) * 0.9,
        )
    box = bands["scene"]
    viewport = Image.fromarray(np.asarray(scene).astype("uint8")).convert("RGB")
    viewport = viewport.resize(
        (box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS
    )
    image.paste(viewport, box[:2])
    text(scale_label, width * 0.5, height * 0.805, 21, "colorbar", center=True)
    x0, x1 = round(width * 0.28), round(width * 0.72)
    y0, y1 = round(height * 0.85), round(height * 0.87)
    gradient = np.linspace(0, 1, x1 - x0)[None, :]
    rgb = np.uint8(colormaps[cmap](gradient)[..., :3] * 255)
    image.paste(Image.fromarray(rgb).resize((x1 - x0, y1 - y0)), (x0, y0))
    for value, fraction in zip(np.linspace(*limits, 5), np.linspace(0, 1, 5)):
        text(
            f"{value:.2f}",
            x0 + (x1 - x0) * fraction,
            height * 0.88,
            17,
            "colorbar",
            center=True,
            max_width=width * 0.09,
        )
    footer = (
        footer or r"Axisymmetric shell reconstruction | $r\in[0.3,1]$ | dimensionless"
    )
    text(footer, width * 0.5, height * 0.948, 17, "footer", center=True)
    for i, first in enumerate(measurements):
        a = first["bbox"]
        for second in measurements[i + 1 :]:
            b = second["bbox"]
            if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
                raise ValueError("Measured annotation boxes overlap")
    return np.asarray(image), {"size": size, "regions": bands, "text": measurements}
