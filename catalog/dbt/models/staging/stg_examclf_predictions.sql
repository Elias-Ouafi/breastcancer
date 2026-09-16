-- Out-of-fold patient scores of imaging.examclf (models/examclf/cv_predictions.csv).
SELECT
    trim(patient) AS patient_id,
    label,
    score,
    fold
FROM {{ source('raw', 'examclf_predictions') }}
