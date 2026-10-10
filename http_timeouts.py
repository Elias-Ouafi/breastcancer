from __future__ import annotations

import logging

log = logging.getLogger(__name__)

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
