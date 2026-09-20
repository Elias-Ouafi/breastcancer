"""Load the YAML parameters, and tell which cases a change of them concerns.

PyYAML is imported inside :func:`load`, not at module level, so that ``steps`` and the
tests of the pure functions never need it.
"""
from __future__ import annotations

import hashlib
import json
import os

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

# The sections each stage depends on. A case is redone only when the hash of *its* sections
# changed, so re-tuning the crop margin does not redo the (slow) DICOM ingestion.
INGEST_SECTIONS = ("ingest",)
PROCESS_SECTIONS = ("channels", "reference_channel", "n4", "body_mask", "registration",
                    "resample", "spacing", "normalize", "crop", "pseudo_mask", "seed")


def load(path=None):
    """The parameters as a plain dict."""
    import yaml

    with open(path or DEFAULT_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def section_hash(settings, sections, extra=None):
    """Short, stable hash of the named sections (plus ``extra``, e.g. the resolved spacing)."""
    payload = {name: settings[name] for name in sections}
    if extra is not None:
        payload["_extra"] = extra
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
