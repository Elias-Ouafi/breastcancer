# 0008 — The demo runs from a clone: versioned, tested artefacts and a narrow image

**Status:** accepted · **In effect:** 2026-08-18

## Context

The demo is what people actually run, often on an unfamiliar machine and minutes before
a meeting. Training needs CUDA, a GPU and more than 100 GB of DICOM, and none of that
should stand between a reviewer and seeing the app work.

## Decision

- The served checkpoints and three curated 25-slice cases (~4.5 MB each) are **versioned
  in git**, through explicit `.gitignore` exceptions.
- Tests check that git **tracks** them, not just that they exist on disk. A file present
  locally but ignored by git is exactly what breaks on a fresh clone.
- `run_demo.py --check` preflights the checkpoint and the cases without binding a port.
- The Docker image installs only the demo's dependencies: CPU PyTorch, Flask, NumPy and
  Pillow, with no JVM, ITK or TCIA client. It runs `read_only` as a non-root user, with
  a healthcheck on `--check`. The port is published on `127.0.0.1` only.
- Each demo case pins its slice (`forced_slice`), and the app reports
  `slice_preselected: true` rather than implying the model found it.

## Consequences

- `git clone` + `pip install` + `python run_demo.py`, or `docker compose up`, is enough.
- The repository carries ~46 MB of versioned model and data files, on purpose.
- The demo cases derive from Duke-Breast-Cancer-MRI (CC BY-NC 4.0): they are shared
  with attribution, for non-commercial use only.
