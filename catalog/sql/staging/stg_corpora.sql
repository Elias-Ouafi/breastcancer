-- What each preprocessing run wrote, from its manifest.json (lineage.py).
--
-- `corpus` is 'dbt' (lesion-cropped, annotated series only) or 'dbt_exams' (every
-- labelled series at 384x384, the exam-level corpus). `case_id` is the PatientID.
CREATE OR REPLACE VIEW stg.corpus_cases AS
SELECT
    corpus,
    series_uid,
    case_id                         AS patient_id,
    study_uid,
    lower(view)                     AS view,
    label,
    exam_status,
    lesion_class,
    n_slices,
    height,
    width,
    lesion_slices,
    lesion_slice_fraction,
    coalesce(mirrored, false)       AS mirrored,
    n_warnings
FROM raw.manifest_cases;

CREATE OR REPLACE VIEW stg.corpus_runs AS
SELECT
    corpus,
    TRY_CAST(generated_at AS TIMESTAMPTZ) AS generated_at,
    git_revision,
    output_dir,
    n_cases,
    n_validation_warnings
FROM raw.manifest_runs;
