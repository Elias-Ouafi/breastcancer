-- Every downloaded series folder is listed by the inventory
SELECT d.series_uid, d.n_files, d.n_bytes
FROM {{ ref('stg_disk_series') }} AS d
ANTI JOIN {{ ref('stg_dbt_file_paths') }} AS fp
    USING (series_uid)
