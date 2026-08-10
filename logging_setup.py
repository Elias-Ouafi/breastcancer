"""One place to configure logging for the pipelines.

The pipelines used to report progress with ``print``, which works right up until you
need the thing every long run needs: a timestamp on each line, a severity you can
filter, and a file you can read after the run instead of a terminal you have already
closed. A 60 GB download or a multi-hour training run is exactly that case.

Two rules, standard but worth stating because breaking either is what makes logging
annoying rather than useful:

* **Modules never configure.** They do ``log = logging.getLogger(__name__)`` and
  nothing else. Configuring at import time would hijack logging for anything that
  imports them -- including pytest and the Flask app.
* **Entry points configure once**, by calling :func:`setup_logging` in ``__main__``.

Level comes from ``$BREASTCANCER_LOG_LEVEL`` when set, so a noisy run can be turned up
without editing code.
"""
from __future__ import annotations

import logging
import os

import config

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


def setup_logging(level=None, logfile=None, force=False):
    """Configure root logging for a pipeline run. Safe to call more than once.

    ``logfile`` is a name inside ``logs/`` (not a full path), so runs cannot scatter
    files across the repo. Pass one for anything long enough that you will want to
    read it afterwards.
    """
    global _configured
    if _configured and not force:
        return logging.getLogger()

    if level is None:
        level = os.environ.get("BREASTCANCER_LOG_LEVEL", "INFO").upper()

    handlers = [logging.StreamHandler()]
    if logfile:
        log_dir = os.path.join(config.ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, logfile), encoding="utf-8"))

    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATE_FORMAT,
                        handlers=handlers, force=True)

    # These two are chatty at INFO and drown out the pipeline's own output.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _configured = True
    return logging.getLogger()
