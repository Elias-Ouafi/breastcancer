# 0005 — Both classes from one source, through one geometry

**Status:** accepted · **In effect:** 2026-09-13 · **Journal:** "Cible chiffrée et voie retenue", §4.7

## Context

An exam-level classifier needs exams **with** and **without** cancer. The easy ways to
get both each leak the label into something that is not the image:

- **Two sources.** Cancers from Duke-Breast-Cancer-MRI (100 % prevalence) plus normals
  from elsewhere teach the model the scanner, protocol and site. The AUC is excellent
  and worthless.
- **Two geometries.** Positives cropped around the lesion (45×72×70) next to full-frame
  negatives (2457×1890) are separable by array shape alone.
- **Two code paths.** If only positives go through the laterality flip, "was flipped"
  becomes a proxy for "has a box", and so for the label.

## Decision

- Both classes come from **Breast-Cancer-Screening-DBT**, a screening collection that is
  mostly normal (4,581 of 5,060 patients).
- The label comes from `BCS-DBT-labels-*.csv`, the only table that says an exam is
  normal. A patient is read at their **worst view**, and a series with no labels row
  is skipped and counted, not assumed normal.
- `preprocess_dbt_exams` sends **every** exam through one geometry: the full frame
  resampled to 384×384 with its aspect ratio kept, zero-padded, every slice kept, no
  cropping. Downsampling averages rather than samples, and the flip applies to
  negatives too.

## Consequences

- Corpus: 870 exams, 272 patients, 56 cancers, 0 validation warnings. ~4.9 MB per series
  compressed, ~14 s to decode each, and resumable.
- The classifier trained on it scores AUC 0.457, which is chance. That result is
  trustworthy precisely because the shortcuts above are closed. A leaky corpus would
  have reported success.
