-- The exam classifier's out-of-fold score, by the patient's true status. A working
-- model ranks cancer above the rest; this one does not, which is ROC-AUC 0.457 read
-- in a form anyone can check (DOCUMENTATION.md §4.11).
SELECT
    status,
    count(*)                            AS patients,
    round(avg(examclf_score), 3)        AS mean_score,
    round(median(examclf_score), 3)     AS median_score,
    round(quantile_cont(examclf_score, 0.9), 3) AS p90_score
FROM mart.dim_patient
WHERE examclf_score IS NOT NULL
GROUP BY status
ORDER BY mean_score DESC;
