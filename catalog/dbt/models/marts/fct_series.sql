-- One row per series the collection publishes: what the tables say about it, whether
-- it is on disk, and which corpus it made it into.
--
-- Grain: series_uid (unique and not null; tested in _marts.yml).
WITH boxes AS (
    SELECT
        patient_id, study_uid, view,
        count(*)                                        AS n_boxes,
        count(*) FILTER (WHERE lesion_class = 'cancer') AS n_cancer_boxes
    FROM {{ ref('stg_dbt_boxes') }}
    GROUP BY ALL
),
lesion_corpus AS (
    SELECT * FROM {{ ref('stg_corpus_cases') }} WHERE corpus = 'dbt'
),
exam_corpus AS (
    SELECT * FROM {{ ref('stg_corpus_cases') }} WHERE corpus = 'dbt_exams'
)
SELECT
    fp.series_uid,
    fp.patient_id,
    fp.study_uid,
    fp.view,
    fp.laterality,
    fp.view_position,
    fp.split,
    l.status                               AS view_status,
    coalesce(b.n_boxes, 0)                 AS n_boxes,
    coalesce(b.n_cancer_boxes, 0)          AS n_cancer_boxes,
    d.series_uid IS NOT NULL               AS on_disk,
    d.n_bytes                              AS disk_bytes,
    lc.series_uid IS NOT NULL              AS in_lesion_corpus,
    ec.series_uid IS NOT NULL              AS in_exam_corpus,
    ec.label                               AS exam_label,
    coalesce(ec.n_slices, lc.n_slices)     AS n_slices,
    coalesce(ec.mirrored, lc.mirrored)     AS mirrored
FROM {{ ref('stg_dbt_file_paths') }} AS fp
LEFT JOIN {{ ref('stg_dbt_labels') }} AS l
    USING (patient_id, study_uid, view)
LEFT JOIN boxes AS b
    USING (patient_id, study_uid, view)
LEFT JOIN {{ ref('stg_disk_series') }} AS d
    ON d.series_uid = fp.series_uid
LEFT JOIN lesion_corpus AS lc
    ON lc.series_uid = fp.series_uid
LEFT JOIN exam_corpus AS ec
    ON ec.series_uid = fp.series_uid
