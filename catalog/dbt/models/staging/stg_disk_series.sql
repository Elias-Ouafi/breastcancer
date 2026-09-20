-- The series folders still held in bronze (data/bronze/tcia/). Bronze is a transit
-- zone: a series is deleted once it is in silver, so this is what has not been
-- promoted yet, not what was ever downloaded (fct_series.on_disk is that).
SELECT series_uid, n_files, n_bytes
FROM {{ source('raw', 'disk_series') }}
