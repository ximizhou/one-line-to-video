"""Deterministic seed-frame generation for image-conditioned local video models.

H3's workbench currently requires an input image (``init_image`` or
``ref_images``).  This module keeps that requirement out of the LangGraph node:
for a provider that supports text-to-video the generated PNG is simply ignored,
while local image-conditioned providers can reuse it.  It is intentionally a
small, replaceable seam; Qwen-Image or a user-uploaded frame can replace this
later without changing the graph.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for candidate in (
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _size_for(aspect_ratio: str) -> tuple[int, int]:
    return (720, 1280) if aspect_ratio == "9:16" else (1280, 720)


def ensure_seed_frame(
    path: str | Path,
    *,
    prompt: str,
    order: int,
    aspect_ratio: str = "9:16",
) -> Path:
    """Create a readable, deterministic storyboard card if *path* is absent.

    This is not presented as a replacement for an image model.  It is a reliable
    bridge for local video models that need a starting frame, and makes the
    project usable before Qwen-Image or a user-uploaded style frame is wired in.
    """
    output = Path(path)
    if output.exists() and output.stat().st_size > 0:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = _size_for(aspect_ratio)
    digest = hashlib.sha256(f"{order}:{prompt}".encode("utf-8")).digest()
    hue = digest[0]
    bg = (max(150, 185 + hue // 8), max(190, 220 - hue // 10), 248)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    # Soft bands make motion from the card visually legible even when a model
    # barely moves small line art.
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = (
            int(bg[0] * (1 - ratio) + 235 * ratio),
            int(bg[1] * (1 - ratio) + 246 * ratio),
            int(bg[2] * (1 - ratio) + 255 * ratio),
        )
        draw.line((0, y, width, y), fill=color)

    margin = int(width * 0.07)
    draw.rounded_rectangle(
        (margin, margin, width - margin, int(height * 0.18)),
        radius=28,
        fill=(255, 255, 255),
        outline=(43, 115, 163),
        width=5,
    )
    draw.text(
        (margin + 28, margin + 24),
        f"STORYBOARD · 镜头 {order:02d}",
        fill=(22, 83, 120),
        font=_font(max(22, width // 30)),
    )

    # A stable abstract “world” motif.  The prompt is printed in a small card so
    # the generated clip remains inspectable even with a non-semantic seed.
    horizon = int(height * 0.63)
    draw.rectangle((0, horizon, width, height), fill=(31, 142, 184))
    for x in range(-20, width + 80, 120):
        draw.arc((x, horizon + 35, x + 90, horizon + 65), 180, 360, fill=(140, 232, 245), width=4)
        draw.arc((x + 40, horizon + 115, x + 130, horizon + 145), 180, 360, fill=(140, 232, 245), width=4)

    points = [
        (0, horizon),
        (width // 6, int(height * 0.47)),
        (width // 3, int(height * 0.58)),
        (width // 2, int(height * 0.42)),
        (width * 2 // 3, int(height * 0.56)),
        (width, int(height * 0.45)),
        (width, horizon),
    ]
    draw.polygon(points, fill=(71, 153, 91), outline=(35, 105, 69))

    sun_r = max(42, width // 11)
    sun_x, sun_y = width - margin - sun_r, int(height * 0.27)
    draw.ellipse(
        (sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r),
        fill=(255, 210, 55),
        outline=(242, 162, 35),
        width=5,
    )
    for angle in range(0, 360, 45):
        import math

        x1 = sun_x + int((sun_r + 18) * math.cos(math.radians(angle)))
        y1 = sun_y + int((sun_r + 18) * math.sin(math.radians(angle)))
        x2 = sun_x + int((sun_r + 42) * math.cos(math.radians(angle)))
        y2 = sun_y + int((sun_r + 42) * math.sin(math.radians(angle)))
        draw.line((x1, y1, x2, y2), fill=(244, 167, 32), width=5)

    # Three cloud bubbles and a flowing arrow keep the visual vocabulary
    # consistent across scenes without claiming the prompt is an image model.
    for cx, cy in ((width // 5, int(height * 0.28)), (width // 2, int(height * 0.34)), (width * 3 // 4, int(height * 0.39))):
        for ox, oy, r in ((-42, 15, 35), (0, -8, 48), (42, 15, 35)):
            draw.ellipse((cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r), fill=(255, 255, 255), outline=(150, 200, 220), width=3)
        draw.rectangle((cx - 65, cy + 12, cx + 65, cy + 42), fill=(255, 255, 255), outline=(150, 200, 220), width=3)

    arrow_x = width // 2
    draw.line((arrow_x, int(height * 0.78), arrow_x + 55, int(height * 0.91)), fill=(116, 225, 242), width=24)
    draw.polygon(
        [(arrow_x + 55, int(height * 0.91)), (arrow_x + 34, int(height * 0.87)), (arrow_x + 73, int(height * 0.87))],
        fill=(116, 225, 242),
    )

    prompt_text = " ".join(prompt.split())
    prompt_text = "\n".join(textwrap.wrap(prompt_text, width=25)[:4])
    card_top = int(height * 0.81)
    draw.rounded_rectangle(
        (margin, card_top, width - margin, height - margin),
        radius=22,
        fill=(255, 255, 255),
        outline=(43, 115, 163),
        width=4,
    )
    draw.text((margin + 22, card_top + 20), prompt_text, fill=(24, 82, 112), font=_font(max(20, width // 34)), spacing=7)
    image.save(output, format="PNG", optimize=True)
    return output
