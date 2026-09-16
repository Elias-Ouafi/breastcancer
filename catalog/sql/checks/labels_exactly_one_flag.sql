-- severity: warn
-- description: Each labels row sets exactly one of Normal / Actionable / Benign / Cancer
SELECT patient_id, study_uid, view, split, n_flags
FROM stg.dbt_labels
WHERE n_flags <> 1
