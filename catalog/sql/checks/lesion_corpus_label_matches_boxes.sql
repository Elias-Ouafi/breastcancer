-- severity: error
-- description: In the lesion corpus, label is 1 exactly when the series carries a cancer box
SELECT c.series_uid, c.patient_id, c.view, c.lesion_class, c.label,
       s.n_boxes, s.n_cancer_boxes
FROM stg.corpus_cases AS c
JOIN mart.fct_series AS s
    USING (series_uid)
WHERE c.corpus = 'dbt'
  AND (s.n_boxes = 0
       OR c.label IS DISTINCT FROM (CASE WHEN s.n_cancer_boxes > 0 THEN 1 ELSE 0 END))
