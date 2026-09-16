-- severity: error
-- description: Box class is benign or cancer (validation.LESION_CLASSES)
SELECT patient_id, study_uid, view, lesion_class
FROM stg.dbt_boxes
WHERE lesion_class IS NULL OR lesion_class NOT IN ('benign', 'cancer')
