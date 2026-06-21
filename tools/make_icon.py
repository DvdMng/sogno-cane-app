"""Generate the SOGNO_CANE app icon (assets/icon.ico + icon.png).

Build-time only (needs Pillow). The runtime app just loads the .ico via QIcon,
so Pillow is NOT a runtime dependency. Run from the project root:

    python tools/make_icon.py
"""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "sogno_cane", "assets")

BG = (10, 4, 24, 255)
PANEL = (20, 8, 40, 255)
PINK = (255, 61, 138)
CYAN = (91, 233, 255)
WHITE = (255, 230, 240)


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def draw(size: int) -> Image.Image:
    # Supersample for smooth curves, then downscale.
    S = max(256, size) * 4
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    r = int(S * 0.22)
    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=r, fill=PANEL)
    d.rounded_rectangle(
        [int(S * 0.012)] * 2 + [int(S * 0.988)] * 2,
        radius=r, outline=PINK, width=max(2, S // 90),
    )

    # Soft radial-ish glow near the top.
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse(
        [int(S * 0.12), int(S * 0.02), int(S * 0.88), int(S * 0.62)],
        fill=(255, 61, 138, 60),
    )
    img.alpha_composite(glow)

    # EEG waveform morphing left->right, coloured pink->cyan.
    n = 600
    pts = []
    for i in range(n + 1):
        x = S * (0.12 + 0.76 * i / n)
        ph = i / n
        amp = S * (0.16) * (0.5 + 0.5 * math.sin(ph * math.pi))
        y = S * 0.52 + amp * math.sin(ph * 9.0 * math.pi)
        pts.append((x, y, ph))
    w = max(3, S // 28)
    for i in range(len(pts) - 1):
        x0, y0, ph = pts[i]
        x1, y1, _ = pts[i + 1]
        col = _lerp(PINK, CYAN, ph)
        d.line([(x0, y0), (x1, y1)], fill=col + (255,), width=w)

    # A note head + stem at the end (music out).
    nx, ny = S * 0.80, S * 0.52
    rr = S * 0.075
    d.ellipse([nx - rr, ny - rr, nx + rr, ny + rr], fill=CYAN + (255,))
    d.line([(nx + rr * 0.8, ny), (nx + rr * 0.8, ny - S * 0.20)],
           fill=WHITE + (255,), width=max(3, S // 40))

    return img.resize((size, size), Image.LANCZOS)


def main() -> int:
    os.makedirs(ASSETS, exist_ok=True)
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [draw(s) for s in sizes]
    ico_path = os.path.join(ASSETS, "icon.ico")
    images[-1].save(
        ico_path, format="ICO",
        sizes=[(s, s) for s in sizes],
    )
    png_path = os.path.join(ASSETS, "icon.png")
    images[-1].save(png_path, format="PNG")
    print(f"wrote {ico_path} and {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
