# 0004 — Match annotations to series by a join, never by inference

**Status:** accepted · **In effect:** 2026-09-13 · **Journal:** §4.4, §4.6

## Context

BCS-DBT publishes lesion boxes in a CSV keyed by patient, study and view. The DICOM
series on disk must be matched to those rows. Two earlier versions inferred the view,
and the measurements are why they are gone:

| Approach | Measured outcome |
|---|---|
| DICOM laterality tag | Read `L` on **all 262** series. 147 of 253 matched, 23 masks painted on background |
| Laterality inferred from pixels | Right 237/262. Took the box of the *other* acquisition of a repeated view (`lmlo` vs `lmlo1`) 4 times, a case no pixel can decide |

## Decision

Read the collection's own inventory, `BCS-DBT-file-paths-*.csv`, which lists
`(PatientID, StudyUID, View)` for every series folder. A box belongs to the series that
carries all three values: a **join**. A folder missing from the inventory is skipped
and counted, not guessed. Pixels are used for one thing only: detecting a study stored
mirrored relative to its boxes, and the manifest records which ones were.

## Consequences

- Rebuilt corpus: 260 series matched (was 253), 0 empty masks, 0 validation warnings.
  Exactly the 4 predicted mismatches moved, and 11 repeated views were matched for the
  first time.
- The inventory tables become a hard dependency, so `download_dbt_tables()` fetches all
  9 tables from a clone.
- General rule adopted: when the source publishes the key, join on it. Inference is a
  fallback to be measured, not a default.
