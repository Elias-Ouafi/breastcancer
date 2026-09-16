-- How much of the collection each stage reached, by split and status: published,
-- downloaded, preprocessed, scored. The funnel a data request starts from.
SELECT
    split,
    coalesce(status, 'unlabelled')                          AS status,
    count(*)                                                AS patients,
    count(*) FILTER (WHERE n_series_on_disk > 0)            AS patients_on_disk,
    count(*) FILTER (WHERE n_series_in_exam_corpus > 0)     AS patients_in_exam_corpus,
    count(*) FILTER (WHERE examclf_score IS NOT NULL)       AS patients_scored,
    round(sum(disk_gb), 1)                                  AS disk_gb
FROM {{ ref('dim_patient') }}
GROUP BY ALL
ORDER BY split, status
