-- Every scored patient has at least one series in the exam corpus
SELECT pr.patient_id, pr.label, pr.fold
FROM {{ ref('stg_examclf_predictions') }} AS pr
ANTI JOIN (SELECT DISTINCT patient_id FROM {{ ref('stg_corpus_cases') }} WHERE corpus = 'dbt_exams') AS c
    USING (patient_id)
