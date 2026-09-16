-- Every inventoried series has a labels row; one without is skipped by preprocess_dbt_exams
SELECT fp.series_uid, fp.patient_id, fp.study_uid, fp.view, fp.split
FROM {{ ref('stg_dbt_file_paths') }} AS fp
ANTI JOIN {{ ref('stg_dbt_labels') }} AS l
    USING (patient_id, study_uid, view)
