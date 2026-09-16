# 0006 — Patient-level evaluation, out-of-fold operating threshold

**Status:** accepted · **In effect:** 2026-09-15 · **Journal:** "Cible chiffrée", §4.3, §4.11

## Context

Slices from one patient are correlated, and so are exams from one patient. Metrics that
resample slices give confidence intervals that are too narrow, and splits that ignore
patients leak. An AUC is also not a decision: a screening tool answers at one threshold.
The reference pair is the French national programme's (sensitivity 82.8 %, specificity
91.4 %).

## Decision

- Splits and cross-validation folds are made **by patient**. A patient's score is the
  max over their exams.
- 95 % confidence intervals are bootstrapped by **resampling patients**.
- The operating threshold is set at the target **sensitivity**, never chosen to maximise
  accuracy, and it is taken **out of fold**: each patient is judged by a threshold fitted
  on the other four folds ([imaging/oppoint.py](../../imaging/oppoint.py)).
- The naive in-sample figure is still published, labelled as not citable, so the gap is
  visible.
- A PPV is never quoted without the prevalence it was measured at.

## Consequences

- Published operating point for `examclf`: sensitivity 78.6 % [67.2–88.9], specificity
  20.4 % [15.3–25.9], PPV 20.4 % at a prevalence of 20.6 %. The answer carries no
  information, and the report says so.
- `oppoint` reads stored predictions and retrains nothing, so there is no threshold to
  retry until the number improves.
- A test checks that no threshold ever sees the patients it judges.
