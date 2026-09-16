# 0003 — Orchestrate with Prefect; idempotent stages; retry only transient failures

**Status:** accepted · **In effect:** by 2026-08-18 · **Journal:** §4.10 (download incident)

## Context

The DCE-MRI stages are slow and uneven: hours of network I/O for the download, CPU-bound
preprocessing over ~200 patients, hours of GPU training. A shell script that dies during
stage three has thrown away stages one and two, and does not say which patient broke.

## Decision

- [pipelines/dce_mri.py](../../pipelines/dce_mri.py) wires
  `download → preprocess → train → evaluate` as a Prefect flow.
- **Idempotence:** each stage checks its own output and skips work already done.
  `--from <stage>` resumes, `--force` redoes, and `--dry-run` prints the plan without
  running anything.
- **Retries only where failure is transient.** A TCIA fetch that times out is retried.
  A crashed training run is not: retrying spends another hour reaching the same
  exception.
- Tasks are thin wrappers around the functions the manual commands call, so the flow
  and the commands cannot drift apart.

## Consequences

- A run that crashes in training resumes at training.
- Prefect is an optional extra (`.[orchestration]`), so the demo install does not pull
  in an orchestrator.
- Known gaps:
  - The DBT branch is not wired into a flow yet.
  - `tcia_utils.nbia` sets no network timeout. On 2026-09-14 a download hung for
    2 h 30 with no error, and a retry cannot catch a request that never fails.
