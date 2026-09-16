-- One row per preprocessed corpus: its size, class balance and the run that wrote it.
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
FROM {{ ref('stg_corpus_cases') }} AS c
LEFT JOIN {{ ref('stg_corpus_runs') }} AS r
    USING (corpus)
GROUP BY c.corpus
ORDER BY c.corpus
