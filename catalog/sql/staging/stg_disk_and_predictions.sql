-- The series folders actually downloaded under data/raw_data/tcia/.
CREATE OR REPLACE VIEW stg.disk_series AS
SELECT series_uid, n_files, n_bytes
FROM raw.disk_series;

-- Out-of-fold patient scores of imaging.examclf (models/examclf/cv_predictions.csv).
CREATE OR REPLACE VIEW stg.examclf_predictions AS
SELECT
    trim(patient) AS patient_id,
    label,
    score,
    fold
FROM raw.examclf_predictions;
