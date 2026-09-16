-- severity: warn
-- description: Every downloaded series folder is listed by the inventory
SELECT d.series_uid, d.n_files, d.n_bytes
FROM stg.disk_series AS d
ANTI JOIN stg.dbt_file_paths AS fp
    USING (series_uid)
