-- Is the exam corpus balanced across views and sides? A class concentrated in one view
-- or one breast is a shortcut a model can learn instead of the lesion (ADR 0005).
SELECT
    view_position,
    laterality,
    count(*)                                        AS series,
    count(*) FILTER (WHERE exam_label = 1)          AS cancer_series,
    round(100.0 * count(*) FILTER (WHERE exam_label = 1) / count(*), 1) AS cancer_pct,
    round(avg(n_slices), 1)                         AS mean_slices,
    count(*) FILTER (WHERE mirrored)                AS mirrored
FROM mart.fct_series
WHERE in_exam_corpus
GROUP BY ALL
ORDER BY ALL;
