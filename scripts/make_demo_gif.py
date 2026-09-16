"""Record the animated demo shown at the top of the README (``docs/img/demo.gif``).

    pip install playwright && python -m playwright install chromium
    python scripts/make_demo_gif.py                  # starts the demo, records, stops it
    python scripts/make_demo_gif.py --url http://127.0.0.1:5000   # an already-running demo

A screen recording made by hand drifts from the app the first time a template changes,
and nobody remembers how it was made. This script replays the walkthrough from
the demo walkthrough of DOCUMENTATION.md in a headless browser -- home page, one-click case 1, the result, a
sweep through the slab with the slider, the MIP view -- and assembles the frames with
Pillow, so the GIF is rebuilt with one command.

Playwright is not a project dependency: only this script needs it, and the demo, the
tests and CI never do.
"""
from __future__ import annotations

import argparse
import io
import os
import socket
import subprocess
import sys
import time
import urllib.request

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "docs", "img", "demo.gif")

VIEWPORT = {"width": 1180, "height": 820}
GIF_WIDTH = 760           # README column width; keeps the file a few MB
SWEEP_FRAMES = 14         # slider positions shown out of the 25-slice slab
SWEEP_MS = 110            # per sweep frame


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_up(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"the demo did not answer at {url} within {timeout_s:.0f} s")


def _frame(page) -> Image.Image:
    png = page.screenshot(type="png")
    img = Image.open(io.BytesIO(png)).convert("RGB")
    height = round(img.height * GIF_WIDTH / img.width)
    return img.resize((GIF_WIDTH, height), Image.LANCZOS)


def record(base_url: str, out_path: str) -> None:
    from playwright.sync_api import sync_playwright

    frames: list[tuple[Image.Image, int]] = []   # (image, duration in ms)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)

        page.goto(base_url, wait_until="networkidle")
        frames.append((_frame(page), 1800))

        page.locator("#demo-shortcuts button").first.click()
        page.wait_for_load_state("networkidle")
        frames.append((_frame(page), 2600))

        slider = page.locator("#slice-range")
        slider.scroll_into_view_if_needed()
        page.mouse.wheel(0, 180)
        page.wait_for_timeout(300)
        best = int(slider.input_value())
        last = int(slider.get_attribute("max"))
        frames.append((_frame(page), 1400))

        # Sweep the whole slab and come back to the evaluated slice: the lesion appears,
        # peaks and fades, which is the moment DOCUMENTATION.md's demo walkthrough says convinces.
        positions = [round(i * last / (SWEEP_FRAMES - 1)) for i in range(SWEEP_FRAMES)]
        for pos in positions + [best]:
            slider.evaluate(
                "(el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }", pos
            )
            page.wait_for_timeout(40)
            frames.append((_frame(page), SWEEP_MS))
        frames[-1] = (frames[-1][0], 1600)

        page.locator("#mip-toggle").click()
        page.wait_for_timeout(200)
        frames.append((_frame(page), 2600))

        browser.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # One shared palette keeps colours from flickering between frames.
    palette = frames[1][0].quantize(colors=128, method=Image.MEDIANCUT)
    images = [img.quantize(palette=palette, dither=Image.NONE) for img, _ in frames]
    images[0].save(
        out_path, save_all=True, append_images=images[1:],
        duration=[d for _, d in frames], loop=0, optimize=True,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"wrote {os.path.relpath(out_path, ROOT)}: {len(images)} frames, {size_mb:.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="record an already-running demo instead of starting one")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.url:
        record(args.url.rstrip("/") + "/", args.out)
        return 0

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "run_demo.py"), "--port", str(port)], cwd=ROOT
    )
    try:
        url = f"http://127.0.0.1:{port}/"
        _wait_until_up(url)
        record(url, args.out)
    finally:
        server.terminate()
        server.wait(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
