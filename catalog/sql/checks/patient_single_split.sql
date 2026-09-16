-- severity: error
-- description: A patient belongs to one split only; otherwise train and test would share a patient
SELECT patient_id, list(DISTINCT split ORDER BY split) AS splits
FROM stg.dbt_file_paths
GROUP BY patient_id
HAVING count(DISTINCT split) > 1
