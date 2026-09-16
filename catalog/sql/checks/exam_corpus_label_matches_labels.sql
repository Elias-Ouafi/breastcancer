-- severity: error
-- description: In the exam corpus, exam_status is the view's worst flag and label is 1 exactly for cancer
SELECT c.series_uid, c.patient_id, c.view, c.exam_status, c.label,
       l.status AS labels_status
FROM stg.corpus_cases AS c
LEFT JOIN stg.dbt_labels AS l
    USING (patient_id, study_uid, view)
WHERE c.corpus = 'dbt_exams'
  AND (c.exam_status IS DISTINCT FROM l.status
       OR c.label IS DISTINCT FROM (CASE WHEN l.status = 'cancer' THEN 1 ELSE 0 END))
