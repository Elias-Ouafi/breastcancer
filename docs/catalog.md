# Metadata catalogue

A single DuckDB file that puts together everything *about* the DBT data: the
collection's 9 annotation tables, which series are on disk, what each preprocessed
corpus holds, and how the exam classifier scored each patient. It is queried in SQL.
Its transformations are a **dbt project**, and each build runs 36 dbt tests on it.

```bash
pip install -e ".[catalog]"        # duckdb + dbt-duckdb
python -m catalog build            # ~10 s on the full collection
python -m catalog checks --all
python -m catalog query "SELECT status, count(*) FROM mart.dim_patient GROUP BY ALL"
python -m catalog query --file catalog/queries/01_data_funnel.sql
python -m catalog docs             # dbt documentation site, lineage graph included
```

The catalogue is written to `data/curated_data/catalog/catalog.duckdb` (7.6 MB). Each
mart table is also exported to Parquet under `parquet/`, and dbt's artefacts (manifest,
run results, docs site) go to `dbt_target/`. The whole folder is derived and gitignored:
deleting it loses nothing.

## Why it exists

Before the catalogue, "how many cancer patients of the validation split are downloaded,
and are they all in the exam corpus?" meant loading three CSVs, two manifests and a
directory listing into pandas and writing the join again. Each answer was a one-off
script, and the joins were never checked. Now it is a single query against tables whose
consistency is tested at every build. Why DuckDB: [ADR 0009](adr/0009-duckdb-metadata-catalogue.md).
Why dbt for the transformations: [ADR 0010](adr/0010-dbt-for-the-catalogue-transformations.md).

## How a build runs

| Step | Tool | What happens |
|---|---|---|
| 1. Load `raw` | Python ([catalog/build.py](../catalog/build.py)) | The part dbt cannot do: pool CSVs whose schemas differ, scan the download directory with `stat`, flatten the JSON manifests |
| 2. `dbt run` | dbt ([catalog/dbt/](../catalog/dbt/)) | Builds the `stg` views and the `mart` tables |
| 3. `dbt test` | dbt | Runs 25 generic tests (grain, not null, accepted values, relationships) and 11 singular tests (domain rules), and keeps every test's offending rows in `qa` |
| 4. Record and export | Python | Writes `qa.check_results` and `main.build_info`, exports the marts to Parquet, and moves the file into place |

`dbt run` and `dbt test` run separately on purpose. `dbt build` skips a model whose
upstream test failed, which would hide exactly the tables needed to investigate the
failure.

The build writes into a temporary directory, under the final file name, and moves the
file into place only once every step has succeeded. A failed build therefore leaves the
previous catalogue intact.

**The file name matters.** dbt-duckdb qualifies every view with the database name, which
DuckDB takes from the file stem (`"catalog".raw.dbt_labels`). An earlier version built
into `tmpXXXX.duckdb` and renamed it afterwards: every `stg` view then failed to bind. A
test now covers this.

## Layers

![dbt lineage graph: raw sources feed stg views, which feed fct_series, dim_patient and the singular tests](img/dbt-lineage.png)

*The dbt lineage graph, from `python -m catalog docs`, captured by
[scripts/make_dbt_lineage_png.py](../scripts/make_dbt_lineage_png.py). Green: `raw`
sources. Blue: models and singular tests. The 25 generic tests are declared in YAML and
not drawn.*

| Layer | Content | Defined in |
|---|---|---|
| `raw` | Sources loaded untouched, plus the file each row came from | [catalog/build.py](../catalog/build.py), declared as dbt sources in [_sources.yml](../catalog/dbt/models/staging/_sources.yml) |
| `stg` | 7 views, one per source: trimmed keys, lower-case views, split taken from the file name, worst-flag status | [catalog/dbt/models/staging/](../catalog/dbt/models/staging/) |
| `mart` | 4 tables with a declared and tested grain | [catalog/dbt/models/marts/](../catalog/dbt/models/marts/) |
| `qa` | One table of offending rows per test, plus `check_results` | [catalog/dbt/tests/](../catalog/dbt/tests/), and the YAML files beside the models |

A macro keeps the schema names as they are (`stg`, `mart`, `qa`) instead of dbt's
default `main_stg`. Staging models are aliased (`stg_dbt_labels` is exposed as
`stg.dbt_labels`). Together, these keep the catalogue's interface unchanged from its
pre-dbt version.

## Mart tables

| Table | Grain | Key columns |
|---|---|---|
| `mart.fct_series` | one series of the collection (22,032) | `series_uid`, `patient_id`, `split`, `view_status`, `n_boxes`, `n_cancer_boxes`, `on_disk`, `disk_bytes`, `in_lesion_corpus`, `in_exam_corpus`, `exam_label`, `n_slices`, `mirrored` |
| `mart.dim_patient` | one patient (5,060) | `status` (worst over all views), `split`, `n_series`, `n_boxes`, `n_series_on_disk`, `fully_on_disk`, `disk_gb`, `n_series_in_exam_corpus`, `examclf_score`, `examclf_fold` |
| `mart.collection_coverage` | split × status | patients published, on disk, in the exam corpus, scored |
| `mart.corpus_summary` | preprocessed corpus | series, patients, positive patients, mean depth, mirrored, git revision of the run |

## Data-quality tests

Each test is a query that returns the rows violating it, so zero rows means a pass. An
`error` test guards a figure that would otherwise be wrong: a label, a key, a split. A
`warn` test flags something the pipeline already tolerates. If an `error` test fails,
`build` and `checks` exit with status 1, but the catalogue is still written so the
failure can be investigated (`SELECT * FROM qa.<test name>`).

**Generic tests (25, all `error`)**, declared in
[_staging.yml](../catalog/dbt/models/staging/_staging.yml) and
[_marts.yml](../catalog/dbt/models/marts/_marts.yml):

| Kind | Where | Guards against |
|---|---|---|
| `unique` + `not_null` | `fct_series.series_uid`, `dim_patient.patient_id`, `stg_dbt_file_paths.series_uid`, (patient, study, view) of `stg_dbt_labels`, (corpus, series) of `stg_corpus_cases`, … | a fanned-out join or a broken grain |
| `accepted_values` | split, status, laterality, view position, lesion class, corpus, exam label | a value no downstream rule knows how to handle |
| `relationships` | `fct_series.patient_id` → `dim_patient` | a series whose patient is missing |

**Singular tests (11)**, in [catalog/dbt/tests/](../catalog/dbt/tests/), with their
severity and description in [_checks.yml](../catalog/dbt/tests/_checks.yml):

| Test | Severity | Guards against |
|---|---|---|
| `boxes_match_inventory` | error | a box that belongs to no series ([ADR 0004](adr/0004-match-annotations-by-join.md)) |
| `patient_single_split` | error | one patient in two splits, i.e. train/test leakage |
| `corpus_cases_match_inventory` | error | a preprocessed volume attached to the wrong patient, study or view |
| `exam_corpus_label_matches_labels` | error | an exam label that disagrees with the labels table |
| `lesion_corpus_label_matches_boxes` | error | a lesion label that disagrees with the box class |
| `predictions_patients_in_exam_corpus` | error | a score for a patient the classifier never saw |
| `predictions_label_matches_corpus` | error | a scored label that differs from the corpus label |
| `labels_exactly_one_flag` | warn | a labels row with zero or several flags |
| `file_paths_have_labels` | warn | a series without a labels row (skipped by preprocessing) |
| `corpus_series_on_disk` | warn | a preprocessed case whose raw series has been deleted |
| `disk_series_in_inventory` | warn | a downloaded folder the inventory does not list |

**On the real data, all 36 pass.** Tests that always pass could simply be unable to
fail, so [tests/test_catalog.py](../tests/test_catalog.py) runs the real dbt project on
a miniature collection and injects defects:
- a wrong exam label;
- a wrong lesion label;
- a case attached to the wrong patient;
- a score for a patient outside the corpus;
- a wrong scored label;
- a duplicated labels key, which must trip the generic `unique` test.

For each one it asserts that the matching test fails. A deleted raw series must raise
only a warning.

## Migration from plain SQL

Before dbt, the same models ran as plain SQL files executed in order. To check that the
move changed nothing, the pre-dbt version was rebuilt from `main` in a temporary
worktree, on the same data. Its four mart tables were then compared with the dbt build,
row by row, using `EXCEPT ALL` in both directions. **All four are identical**: 22,032,
5,060, 12 and 2 rows, with the same columns in the same order. See journal §4.13.

## Example queries

Stored in [catalog/queries/](../catalog/queries/) and run by the test suite, so a
renamed column breaks a test rather than the documentation. Outputs below are from the
build of 2026-09-16.

**1. Data funnel**: published → downloaded → in the exam corpus → scored
(`01_data_funnel.sql`, abridged):

| split | status | patients | on disk | in exam corpus | scored | disk GB |
|---|---|---:|---:|---:|---:|---:|
| train | cancer | 39 | 39 | 39 | 39 | 6.3 |
| train | benign | 62 | 62 | 62 | 62 | 10.5 |
| train | normal | 4,083 | 135 | 135 | 135 | 44.7 |
| validation | cancer | 20 | **17** | 17 | 17 | 2.6 |
| validation | benign | 20 | **14** | 14 | 14 | 2.9 |
| validation | normal | 200 | 5 | 5 | 5 | 1.5 |
| test | cancer | 30 | 30 | **0** | 0 | 5.3 |
| test | benign | 30 | 30 | **0** | 0 | 4.5 |

The status totals over the collection reproduce the figures that were previously taken
from `TransformData.dbt_patient_status`: 4,581 normal, 278 actionable, 112 benign and 89
cancer patients.

**2. Annotated patients never downloaded** (`02_annotated_patients_not_downloaded.sql`):
9 patients, all in the validation split, **3 of them cancers** (DBT-P03621, DBT-P03628,
DBT-P04596). They are the unexplained gap between the 89 cancer patients the collection
publishes and the 86 on disk. They are also 3 cancers the exam classifier could have
trained on.

**3. Classifier score by true status** (`03_examclf_score_by_status.sql`):

| status | patients | mean score | median |
|---|---:|---:|---:|
| normal | 140 | 0.187 | 0.182 |
| cancer | 56 | 0.181 | 0.168 |
| benign | 76 | 0.180 | 0.161 |

Normal patients score *higher* on average than cancer patients. This is the published
AUC of 0.457 (journal §4.11), shown in a form anyone can recompute.

**4. Exam corpus balance by view and side** (`04_exam_corpus_balance.sql`): cancer
prevalence ranges from 9.9 % (left MLO) to 14.4 % (right MLO), and all 48 mirrored
series are left views. That is what the flip rule of
[ADR 0005](adr/0005-one-source-one-geometry.md) implies, and it is worth knowing before
using `mirrored` as a feature.

## Findings the catalogue surfaced

- **3 cancer patients were never downloaded** (query 2). Downloading them costs about
  0.5 GB: 6 series, at the ~85 MB average of the cancer series already on disk.
- **The test split's 60 annotated patients are on disk but in no corpus.** They were
  downloaded on 2026-09-14, after the exam corpus had been built (2026-09-13).
- **152 normal patients are on disk**, while the journal describes 150 downloaded
  normals. Not investigated yet.

## Limits

- DBT only. The DCE-MRI corpus predates `lineage.py` and has no manifest to load.
- Disk sizes come from `stat` and say a series is present, not that its DICOM is
  readable.
- Rebuilding is cheap, so there is no incremental model. It takes 9.7 s, against 3.9 s
  for the plain-SQL version: the price of dbt starting and parsing its project twice
  (three builds each, same data).
