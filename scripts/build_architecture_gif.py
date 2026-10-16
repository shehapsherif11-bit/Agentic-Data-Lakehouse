"""
build_architecture_gif.py - builds the animated architecture diagram of the README.

    python scripts/build_architecture_gif.py            # writes docs/images/architecture.gif and architecture.png

The drawing lives in scripts/arch_gif/template.html (one SVG, a pure function of time: render(t)). This script opens it in
a headless Chrome, takes one screenshot per frame and assembles the GIF with Pillow. Tool logos are the real SVG icons in
scripts/arch_gif/icons (simple-icons and lobehub icons); the font is Poppins (scripts/arch_gif/fonts).

Needs: pip install playwright pillow, and Google Chrome (or run `playwright install chromium` and pass --channel "").
"""
import argparse
import io
import json
import re
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent / "arch_gif"
ICON_FILES = {  # name used in the template -> file in arch_gif/icons
    "streamlit": "streamlit", "langgraph": "langgraph-lb", "groq": "groq-lb", "openrouter": "openrouter-lb",
    "databricks": "databricks", "duckdb": "duckdb", "plotly": "plotly", "dbt": "dbt", "airflow": "airflow", "docker": "docker",
}
BUILD_END, SCAN_START, SCAN_END = 5.0, 5.3, 8.1   # seconds: the diagram builds up, then a highlight scans stages 1 to 7
BUILD_STEP, SCAN_STEP, HOLD_MS = 0.1, 0.2, 3000


def load_icons():
    icons = {}
    for name, stem in ICON_FILES.items():
        svg = (SRC / "icons" / f"{stem}.svg").read_text(encoding="utf-8")
        icons[name] = {
            "vb": re.search(r'viewBox="([^"]+)"', svg).group(1),
            "paths": re.findall(r'<path[^>]*?\sd="([^"]+)"', svg),
            "rule": "evenodd" if 'fill-rule="evenodd"' in svg else "nonzero",
        }
    return icons


def timeline():
    """[(t, scan, duration_ms)] for every frame; the last one is the finished diagram, held before the loop restarts."""
    frames, t = [], BUILD_STEP
    while t <= BUILD_END + 1e-9:
        frames.append((round(t, 3), False, int(BUILD_STEP * 1000)))
        t += BUILD_STEP
    t = SCAN_START
    while t <= SCAN_END + 1e-9:
        frames.append((round(t, 3), True, int(SCAN_STEP * 1000)))
        t += SCAN_STEP
    frames.append((99.0, False, HOLD_MS))
    return frames


def shared_palette(frames, colors=255):
    """One palette for every frame (no flicker, and Pillow can store only the changed pixels of each frame)."""
    picks = [frames[i] for i in range(0, len(frames), max(1, len(frames) // 8))] + [frames[-1]]
    w, h = picks[0].size
    sheet = Image.new("RGB", (w // 2, (h // 2) * len(picks)))
    for i, f in enumerate(picks):
        sheet.paste(f.resize((w // 2, h // 2), Image.LANCZOS), (0, i * (h // 2)))
    return sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "images"))
    ap.add_argument("--width", type=int, default=1080, help="GIF width in px (the page is drawn at 1080 x 1350)")
    ap.add_argument("--channel", default="chrome", help="installed browser to use; '' for Playwright's own Chromium")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    html = (SRC / "template.html").read_text(encoding="utf-8")
    html = html.replace("__ICONS__", json.dumps(load_icons())).replace("__FONT_DIR__", (SRC / "fonts").as_uri())
    plan = timeline()

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
        page_file = Path(tmp) / "arch.html"
        page_file.write_text(html, encoding="utf-8")
        browser = pw.chromium.launch(channel=args.channel or None)

        def shot(t, scan, scale=1):
            page = browser.new_page(viewport={"width": 1080, "height": 1350}, device_scale_factor=scale)
            page.goto(page_file.as_uri())
            page.wait_for_function("window.READY === true")
            page.evaluate(f"render({t}, {'true' if scan else 'false'})")
            png = page.screenshot(type="png")
            page.close()
            return Image.open(io.BytesIO(png)).convert("RGB")

        page = browser.new_page(viewport={"width": 1080, "height": 1350})
        page.goto(page_file.as_uri())
        page.wait_for_function("window.READY === true")
        frames = []
        for t, scan, _ in plan:
            page.evaluate(f"render({t}, {'true' if scan else 'false'})")
            frames.append(Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB"))
        page.close()
        poster = shot(99.0, False, scale=2)
        browser.close()

    if args.width != 1080:
        size = (args.width, round(1350 * args.width / 1080))
        frames = [f.resize(size, Image.LANCZOS) for f in frames]
    pal = shared_palette(frames)
    quant = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    gif = out / "architecture.gif"
    quant[0].save(gif, save_all=True, append_images=quant[1:], duration=[d for *_, d in plan], loop=0, optimize=False)
    poster.save(out / "architecture.png", optimize=True)
    print(f"{gif}  {gif.stat().st_size / 1e6:.2f} MB, {len(frames)} frames")
    print(f"{out / 'architecture.png'}  {(out / 'architecture.png').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
