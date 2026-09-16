-- Every preprocessed case still has its raw series on disk, so it can be rebuilt
SELECT c.corpus, c.series_uid, c.patient_id
FROM {{ ref('stg_corpus_cases') }} AS c
ANTI JOIN {{ ref('stg_disk_series') }} AS d
    USING (series_uid)
