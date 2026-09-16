-- One row per patient: the worst status over all their views, what is downloaded and
-- preprocessed, and the exam classifier's out-of-fold score where there is one.
--
-- Grain: patient_id. `split` is unique per patient in BCS-DBT (checked by
-- qa.patient_single_split); any_value would hide a violation, so the check exists.
CREATE OR REPLACE TABLE mart.dim_patient AS
WITH per_patient AS (
    SELECT
        patient_id,
        any_value(split)                                      AS split,
        count(DISTINCT study_uid)                             AS n_studies,
        count(*)                                              AS n_series,
        -- worst first: the same order TransformData.DBT_STATUS_ORDER applies
        CASE min(CASE view_status
                     WHEN 'cancer' THEN 1 WHEN 'benign' THEN 2
                     WHEN 'actionable' THEN 3 WHEN 'normal' THEN 4 END)
            WHEN 1 THEN 'cancer' WHEN 2 THEN 'benign'
            WHEN 3 THEN 'actionable' WHEN 4 THEN 'normal'
        END                                                   AS status,
        sum(n_boxes)                                          AS n_boxes,
        count(*) FILTER (WHERE on_disk)                       AS n_series_on_disk,
        round(coalesce(sum(disk_bytes), 0) / 1e9, 3)          AS disk_gb,
        count(*) FILTER (WHERE in_lesion_corpus)              AS n_series_in_lesion_corpus,
        count(*) FILTER (WHERE in_exam_corpus)                AS n_series_in_exam_corpus
    FROM mart.fct_series
    GROUP BY patient_id
)
SELECT
    p.*,
    p.n_series_on_disk = p.n_series        AS fully_on_disk,
    pr.score                               AS examclf_score,
    pr.fold                                AS examclf_fold
FROM per_patient AS p
LEFT JOIN stg.examclf_predictions AS pr
    USING (patient_id);
