"""Give a third-party client's HTTP calls a timeout it does not set itself.

`tcia_utils.nbia` (3.3.4) sends every request with ``requests.get(url)`` and no
``timeout``. `requests` then waits forever: on 2026-09-14 a series download hung on one
request for 2 h 30, alive, silent, with no error to retry on.

The obvious fix does not work, and that was measured rather than assumed. With
``socket.setdefaulttimeout(2)``, a ``requests.get`` against a server that accepts and
never answers was still blocked after 8 s: `requests` passes an explicit "no timeout"
down to the socket, which overrides the process default. Only a ``timeout=`` argument
on the call itself raised (``ReadTimeout`` after 2.0 s).

So the client's reference to the ``requests`` module is swapped for a thin proxy that
adds a default ``timeout`` to ``get`` and ``post`` and forwards everything else. The
read timeout bounds *inactivity* between bytes, not the total duration: a large series
that keeps streaming is never cut, a stalled one is.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Calibrated on the download log of the 2026-09-14 run (122 series): a series took
# 13 s at the median, 49 s at p90 and 565 s at worst -- that whole duration being the
# upper bound of any wait between two bytes. 600 s clears the slowest successful series
# even if all of it had been waiting for the first byte, and turns a 2 h 22 stall into a
# 10-minute one that the orchestrator can retry.
CONNECT_TIMEOUT_S = 30
READ_TIMEOUT_S = 600


class RequestsWithTimeout:
    """Stands in for the ``requests`` module; only ``get`` and ``post`` change."""

    def __init__(self, module, timeout):
        self._module = module
        self.timeout = timeout

    def get(self, *args, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        return self._module.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        return self._module.post(*args, **kwargs)

    def __getattr__(self, name):  # exceptions, Session, codes... unchanged
        return getattr(self._module, name)


def install(client_module, connect=CONNECT_TIMEOUT_S, read=READ_TIMEOUT_S):
    """Make every ``requests.get``/``post`` issued by ``client_module`` time out.

    Idempotent: installing twice replaces the timeout instead of stacking proxies.
    Returns the proxy.
    """
    current = client_module.requests
    base = current._module if isinstance(current, RequestsWithTimeout) else current
    proxy = RequestsWithTimeout(base, (connect, read))
    client_module.requests = proxy
    log.debug("HTTP timeouts for %s: connect %ss, read %ss",
              getattr(client_module, "__name__", client_module), connect, read)
    return proxy
