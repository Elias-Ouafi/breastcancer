-- severity: error
-- description: Every box joins to exactly one inventoried series on (patient, study, view)
SELECT b.patient_id, b.study_uid, b.view, b.split, b.lesion_class
FROM stg.dbt_boxes AS b
ANTI JOIN stg.dbt_file_paths AS fp
    USING (patient_id, study_uid, view)
