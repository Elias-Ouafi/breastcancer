"""Capture the dbt lineage graph shown in DOCUMENTATION.md (catalogue section) (``docs/img/dbt-lineage.png``).

    python -m catalog build && python -m catalog docs
    python scripts/make_dbt_lineage_png.py

Serves the generated dbt docs site on a free local port, opens its full lineage graph in
headless Chromium and saves the graph panel. Needs Playwright, like
``make_demo_gif.py``: it is not a project dependency.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import io
import os
import sys
import threading

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402

DEFAULT_TARGET = os.path.join(config.CATALOG_DIR, "dbt_target")
DEFAULT_OUT = os.path.join(ROOT, "docs", "img", "dbt-lineage.png")


def capture(target_dir: str, out_path: str) -> None:
    from playwright.sync_api import sync_playwright

    if not os.path.exists(os.path.join(target_dir, "index.html")):
        sys.exit(f"no dbt docs in {target_dir}; run `python -m catalog docs` first")

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # the docs site requests a few icons it never ships
            pass

    handler = functools.partial(QuietHandler, directory=target_dir)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1500, "height": 900})
            url = f"http://127.0.0.1:{server.server_address[1]}/index.html#!/overview?g_v=1"
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(4000)  # the graph lays itself out after load
            png = page.screenshot()
            browser.close()
    finally:
        server.shutdown()

    # Keep the graph panel, drop the page frame and the selector bar under it.
    image = Image.open(io.BytesIO(png)).crop((20, 20, 1480, 815))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path, optimize=True)
    print(f"wrote {os.path.relpath(out_path, ROOT)} ({os.path.getsize(out_path) // 1024} KB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-dir", default=DEFAULT_TARGET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    capture(args.target_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
