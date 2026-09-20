-- Medallion rule: bronze is a transit zone, so a series a silver corpus already holds
-- should no longer be in bronze. Rows here are the purge still to run (or a
-- --keep-bronze run); warn only, since nothing about the labels is wrong.
SELECT c.corpus, c.series_uid, c.patient_id
FROM {{ ref('stg_corpus_cases') }} AS c
SEMI JOIN {{ ref('stg_disk_series') }} AS d
    USING (series_uid)
