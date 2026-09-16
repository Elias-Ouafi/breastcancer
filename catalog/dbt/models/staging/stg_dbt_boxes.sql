-- One row per annotated lesion box of BCS-DBT-boxes-*.csv, all splits pooled.
--
-- `volume_slices` is NULL for the train split, whose table does not carry the column.
SELECT
    trim(PatientID)                                            AS patient_id,
    trim(StudyUID)                                             AS study_uid,
    lower(trim("View"))                                        AS view,
    regexp_extract(filename, '-(train|validation|test)[-.]', 1) AS split,
    lower(trim(Class))                                         AS lesion_class,
    "Slice"::INTEGER                                           AS slice,
    X::INTEGER                                                 AS x,
    Y::INTEGER                                                 AS y,
    Width::INTEGER                                             AS width,
    Height::INTEGER                                            AS height,
    (AD = 1)                                                   AS architectural_distortion,
    TRY_CAST(VolumeSlices AS INTEGER)                          AS volume_slices
FROM {{ source('raw', 'dbt_boxes') }}
