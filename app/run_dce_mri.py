"""Convenience launcher: runs the Flask app with the DCE-MRI U-Net backend.

Equivalent to setting MRI_APP_BACKEND=dce_mri before `python -m app.server`,
provided here as a single entry point (e.g. for an IDE/launch config).
"""
import os

os.environ.setdefault("MRI_APP_BACKEND", "dce_mri")

from app.server import main

if __name__ == "__main__":
    main()
