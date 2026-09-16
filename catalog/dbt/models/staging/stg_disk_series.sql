-- The series folders actually downloaded under data/raw_data/tcia/.
SELECT series_uid, n_files, n_bytes
FROM {{ source('raw', 'disk_series') }}
