-- One row per (patient, study, view) of BCS-DBT-labels-*.csv, all splits pooled.
--
-- `status` is the view's worst flag, in the order TransformData.DBT_STATUS_ORDER uses
-- (cancer > benign > actionable > normal). A row with no flag set gets NULL rather
-- than a guess: an all-zero row is not evidence that the view is normal.
CREATE OR REPLACE VIEW stg.dbt_labels AS
SELECT
    trim(PatientID)                                            AS patient_id,
    trim(StudyUID)                                             AS study_uid,
    lower(trim("View"))                                        AS view,
    regexp_extract(filename, '-(train|validation|test)[-.]', 1) AS split,
    CASE
        WHEN Cancer = 1     THEN 'cancer'
        WHEN Benign = 1     THEN 'benign'
        WHEN Actionable = 1 THEN 'actionable'
        WHEN Normal = 1     THEN 'normal'
    END                                                        AS status,
    (Normal + Actionable + Benign + Cancer)::INTEGER           AS n_flags
FROM raw.dbt_labels;
