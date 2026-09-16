-- Every preprocessed case is an inventoried series, with the same patient, study and view
SELECT c.corpus, c.series_uid, c.patient_id, c.study_uid, c.view,
       fp.patient_id AS inventory_patient_id, fp.study_uid AS inventory_study_uid,
       fp.view AS inventory_view
FROM {{ ref('stg_corpus_cases') }} AS c
LEFT JOIN {{ ref('stg_dbt_file_paths') }} AS fp
    USING (series_uid)
WHERE fp.series_uid IS NULL
   OR c.patient_id IS DISTINCT FROM fp.patient_id
   OR c.study_uid IS DISTINCT FROM fp.study_uid
   OR c.view IS DISTINCT FROM fp.view
