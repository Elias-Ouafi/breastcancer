-- Annotated patients (they carry a lesion box) with nothing on disk. Every cancer
-- patient here is one the exam classifier could have trained on and never saw, so this
-- is the download to-do list before any "we need more data" discussion.
SELECT
    patient_id,
    split,
    status,
    n_series,
    n_boxes
FROM mart.dim_patient
WHERE n_boxes > 0
  AND n_series_on_disk = 0
ORDER BY status DESC, split, patient_id;
