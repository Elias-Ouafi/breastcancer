-- How much of the collection each stage reached, by split and status: published,
-- downloaded, preprocessed, scored. The funnel a data request starts from.
CREATE OR REPLACE TABLE mart.collection_coverage AS
SELECT
    split,
    coalesce(status, 'unlabelled')                          AS status,
    count(*)                                                AS patients,
    count(*) FILTER (WHERE n_series_on_disk > 0)            AS patients_on_disk,
    count(*) FILTER (WHERE n_series_in_exam_corpus > 0)     AS patients_in_exam_corpus,
    count(*) FILTER (WHERE examclf_score IS NOT NULL)       AS patients_scored,
    round(sum(disk_gb), 1)                                  AS disk_gb
FROM mart.dim_patient
GROUP BY ALL
ORDER BY split, status;

-- One row per preprocessed corpus: its size, class balance and the run that wrote it.
CREATE OR REPLACE TABLE mart.corpus_summary AS
SELECT
    c.corpus,
    count(*)                                                 AS n_series,
    count(DISTINCT c.patient_id)                             AS n_patients,
    count(DISTINCT c.patient_id) FILTER (WHERE c.label = 1)  AS n_positive_patients,
    count(*) FILTER (WHERE c.label = 1)                      AS n_positive_series,
    round(avg(c.n_slices), 1)                                AS mean_slices,
    count(*) FILTER (WHERE c.mirrored)                       AS n_mirrored,
    sum(c.n_warnings)                                        AS n_case_warnings,
    any_value(r.git_revision)                                AS git_revision,
    any_value(r.generated_at)                                AS generated_at
FROM stg.corpus_cases AS c
LEFT JOIN stg.corpus_runs AS r
    USING (corpus)
GROUP BY c.corpus
ORDER BY c.corpus;
