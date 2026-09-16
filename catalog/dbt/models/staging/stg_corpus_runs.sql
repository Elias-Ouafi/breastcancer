-- One row per preprocessing run: when, from which commit, how many cases.
SELECT
    corpus,
    TRY_CAST(generated_at AS TIMESTAMPTZ) AS generated_at,
    git_revision,
    output_dir,
    n_cases,
    n_validation_warnings
FROM {{ source('raw', 'manifest_runs') }}
