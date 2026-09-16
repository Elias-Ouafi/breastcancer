-- severity: error
-- description: Each series folder appears once in the inventory (the grain of mart.fct_series)
SELECT series_uid, count(*) AS n_rows
FROM stg.dbt_file_paths
GROUP BY series_uid
HAVING count(*) > 1
