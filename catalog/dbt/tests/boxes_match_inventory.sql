-- Every box joins to exactly one inventoried series on (patient, study, view)
SELECT b.patient_id, b.study_uid, b.view, b.split, b.lesion_class
FROM {{ ref('stg_dbt_boxes') }} AS b
ANTI JOIN {{ ref('stg_dbt_file_paths') }} AS fp
    USING (patient_id, study_uid, view)
