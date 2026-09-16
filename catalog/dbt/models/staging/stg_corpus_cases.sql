-- What each preprocessing run wrote, one row per case of its manifest.json (lineage.py).
--
-- `corpus` is 'dbt' (lesion-cropped, annotated series only) or 'dbt_exams' (every
-- labelled series at 384x384, the exam-level corpus). `case_id` is the PatientID.
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
FROM {{ source('raw', 'manifest_cases') }}
