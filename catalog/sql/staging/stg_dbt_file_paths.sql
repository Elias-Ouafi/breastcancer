-- The series inventory: one row per series folder the collection publishes.
--
-- `series_uid` is the second-to-last segment of `classic_path`, which is the folder
-- name the downloader writes (TransformData.series_uid_from_classic_path). It is the
-- key that ties the tables to the disk and to the manifests.
CREATE OR REPLACE VIEW stg.dbt_file_paths AS
SELECT
    string_split(replace(classic_path, '\', '/'), '/')[-2]     AS series_uid,
    trim(PatientID)                                            AS patient_id,
    trim(StudyUID)                                             AS study_uid,
    lower(trim("View"))                                        AS view,
    left(lower(trim("View")), 1)                               AS laterality,
    regexp_extract(lower(trim("View")), '^[lr]([a-z]+)', 1)    AS view_position,
    regexp_extract(filename, '-(train|validation|test)[-.]', 1) AS split,
    classic_path
FROM raw.dbt_file_paths;
