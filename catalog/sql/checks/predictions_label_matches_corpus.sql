-- severity: error
-- description: A scored patient's label is the max of their exam-corpus labels
SELECT pr.patient_id, pr.label AS prediction_label, c.corpus_label
FROM stg.examclf_predictions AS pr
JOIN (SELECT patient_id, max(label) AS corpus_label
      FROM stg.corpus_cases WHERE corpus = 'dbt_exams' GROUP BY patient_id) AS c
    USING (patient_id)
WHERE pr.label IS DISTINCT FROM c.corpus_label
