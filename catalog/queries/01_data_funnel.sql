-- How far each part of the collection got: published -> downloaded -> in the exam
-- corpus -> scored by the classifier. The first question of any data request.
SELECT
    split,
    status,
    patients,
    patients_on_disk,
    patients_in_exam_corpus,
    patients_scored,
    bronze_gb
FROM mart.collection_coverage
ORDER BY split, CASE status WHEN 'cancer' THEN 1 WHEN 'benign' THEN 2
                            WHEN 'actionable' THEN 3 ELSE 4 END;
