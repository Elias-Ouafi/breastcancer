"""Where the corpus says what it holds, what it left out, and what it did to each case.

Three files, all plain text, all rewritten atomically:

* ``registry.csv``   one row per patient: scanner, field strength, site, study date and the
  phases available. Duke publishes no site (``InstitutionName`` is empty) and anonymises the
  date to 1990-01-01, so ``site`` is read from the header and left empty when it is: the
  scanner (manufacturer, model, field strength) is the only acquisition-level variable that
  actually varies, and is what the shortcut check would have to stratify on.
* ``exclusions.csv`` one row per patient that did not make it, with the reason. A case is
  never dropped without leaving a line here.
* ``log/cases.jsonl`` one JSON object per processed case: the steps applied, their
  parameters, their durations, and the anomalies met.
"""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone

REGISTRY_COLUMNS = ["patient_id", "scanner", "field_strength_t", "site", "study_date",
                    "phases_available", "series_descriptions", "n_slices", "native_spacing_mm",
                    "status"]
EXCLUSION_COLUMNS = ["patient_id", "stage", "reason", "detail", "timestamp"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    """Write ``rows`` (dicts) to ``path`` atomically: a crash never leaves half a registry."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def upsert(path, columns, key, row):
    """Insert or replace ``row`` in the CSV at ``path``, keyed on ``row[key]``."""
    rows = [r for r in read_csv(path) if r[key] != row[key]]
    rows.append({column: row.get(column, "") for column in columns})
    rows.sort(key=lambda r: r[key])
    write_csv(path, columns, rows)


def drop(path, columns, key, value):
    """Remove the row of ``key == value``; used when a case that was excluded now succeeds."""
    rows = [r for r in read_csv(path) if r[key] != value]
    write_csv(path, columns, rows)


def exclude(out_dir, patient_id, stage, reason, detail=""):
    """Record that ``patient_id`` was left out, and why. Returns the row."""
    row = {"patient_id": patient_id, "stage": stage, "reason": reason,
           "detail": str(detail)[:300], "timestamp": now()}
    upsert(os.path.join(out_dir, "exclusions.csv"), EXCLUSION_COLUMNS, "patient_id", row)
    return row


def clear_exclusion(out_dir, patient_id):
    drop(os.path.join(out_dir, "exclusions.csv"), EXCLUSION_COLUMNS, "patient_id", patient_id)


class CaseLog:
    """Collects the steps of one case, then appends them as one line of ``cases.jsonl``."""

    def __init__(self, out_dir, patient_id, stage):
        self.path = os.path.join(out_dir, "log", "cases.jsonl")
        self.record = {"patient_id": patient_id, "stage": stage, "started": now(),
                       "steps": [], "anomalies": []}
        self._t0 = time.perf_counter()

    def step(self, name, seconds, **details):
        self.record["steps"].append({"step": name, "seconds": round(seconds, 2), **details})

    def anomaly(self, message):
        self.record["anomalies"].append(message)

    def timed(self, name, **static):
        """Context manager: ``with log.timed("n4", channel="pre") as details: details["x"] = 1``.

        ``static`` is recorded with the step from the start; ``details`` collects the rest."""
        return _Timed(self, name, static)

    def close(self, status, **details):
        self.record.update({"status": status, "total_seconds": round(time.perf_counter() - self._t0, 2),
                            **details})
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.record, default=str, ensure_ascii=False) + "\n")
        return self.record


class _Timed:
    def __init__(self, log, name, static):
        self.log, self.name, self.static, self.details = log, name, static, {}

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self.details

    def __exit__(self, exc_type, exc, tb):
        self.log.step(self.name, time.perf_counter() - self._t0, **self.static, **self.details)
        return False
