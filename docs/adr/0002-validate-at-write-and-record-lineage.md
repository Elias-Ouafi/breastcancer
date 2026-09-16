# 0002 — Validate at the single write point, record lineage beside the data

**Status:** accepted · **In effect:** 2026-08-18 · **Journal:** §4.2

## Context

Everything downstream trusts the preprocessed `.npz` files blindly. A malformed volume
does not crash: it trains a model on nonsense, or produces a metric that looks right
and is not. That already happened once. `crop=True` recentred each volume on its
lesion, and the model scored a confidence of 1.0000 on 9 test patients out of 9,
because the task had become trivial.

A folder of `.npz` files also cannot say six weeks later which series, which
parameters or which commit produced it.

## Decision

- **Validation** ([validation.py](../../validation.py)) runs in the one function that
  writes a volume. It checks shape, dtype, finiteness and mask binarity, and fails
  loudly, naming the case.
- Thresholds are calibrated on the data, not guessed. "Lesion fraction too high" was
  measured and rejected: legitimate volumes range from 3.3 % to 50.9 %. The in-plane
  size separates the two cases cleanly: real volumes are 448–512 px, crops were ~45–72 px,
  and the limit sits at 128 px.
- **Lineage** ([lineage.py](../../lineage.py)) writes a `manifest.json` per output
  folder: git revision (suffixed `-dirty` if uncommitted), source, parameters and
  per-case stats. It is written **last**, so a missing manifest means an interrupted
  run.

## Consequences

- A bad volume fails at write time instead of hours into training.
- Class counts can be read from a corpus without opening a single volume.
- This is deliberately not a lineage *system*: no server, no database, no run ID. That
  cost pays off only with several teams and scheduled reruns.
- Known gap: `dce_mri_p2/` predates `lineage.py` and has no manifest.
