-- severity: error
-- description: One labels row per (patient, study, view); a duplicate would fan out every join on that key
SELECT patient_id, study_uid, view, count(*) AS n_rows
FROM stg.dbt_labels
GROUP BY ALL
HAVING count(*) > 1
