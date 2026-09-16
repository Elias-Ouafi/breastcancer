# 0001 — Layered storage, every path defined once

**Status:** accepted · **In effect:** by 2026-08-18

## Context

The same folder name used to be written out in a dozen files: an `argparse` default
here, a function default there, a string in the app. Moving a dataset meant searching
the whole tree. Folder names also described experiments (`results_mri_p2`), so a
directory's name said which week produced it, not what it contained.

## Decision

- Three data layers under `data/`:
  - `raw_data`: bytes as downloaded, never written to again.
  - `preprocessed_data`: normalised volumes and masks, one `.npz` per series.
  - `curated_data`: slice banks, exam banks and demo cases. Derived and cheap to
    rebuild.
- Artefacts (`models/`, `reports/`) are a separate tree from data. `data/` is
  gitignored.
- Every path is defined once, as an absolute path, in [config.py](../../config.py).

## Consequences

- Moving a dataset is a one-line change, and a script behaves the same from any working
  directory.
- Anything in `curated_data` can be deleted and rebuilt. Nothing in `raw_data` should
  ever be modified.
- Known exception: `models/dce_mri_p2_negfix/` still names an experiment. Renaming it
  would break versioned paths the demo depends on, so it stays open in the journal.
