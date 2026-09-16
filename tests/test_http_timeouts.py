"""The timeout added to tcia_utils' HTTP calls, tested without tcia_utils.

The last test reproduces the measurement behind the module against a local server that
accepts connections and never answers: without a timeout the call would hang forever,
with the proxy it must raise within the read timeout.
"""
from __future__ import annotations

import socket
import threading
import time
import types

import pytest

import http_timeouts


def fake_client():
    seen = []
    requests = types.SimpleNamespace(
        get=lambda *a, **kw: seen.append(("get", kw)) or "response",
        post=lambda *a, **kw: seen.append(("post", kw)) or "response",
        exceptions="the exceptions module",
    )
    return types.SimpleNamespace(__name__="fake_client", requests=requests), seen


def test_get_and_post_receive_the_default_timeout():
    client, seen = fake_client()
    http_timeouts.install(client, connect=3, read=7)
    client.requests.get("url")
    client.requests.post("url", data={})
    assert seen == [("get", {"timeout": (3, 7)}), ("post", {"data": {}, "timeout": (3, 7)})]


def test_an_explicit_timeout_is_kept():
    client, seen = fake_client()
    http_timeouts.install(client)
    client.requests.get("url", timeout=1)
    assert seen == [("get", {"timeout": 1})]


def test_everything_else_is_forwarded():
    client, _ = fake_client()
    http_timeouts.install(client)
    assert client.requests.exceptions == "the exceptions module"


def test_installing_twice_replaces_instead_of_stacking():
    client, seen = fake_client()
    http_timeouts.install(client, connect=1, read=1)
    proxy = http_timeouts.install(client, connect=2, read=5)
    assert not isinstance(proxy._module, http_timeouts.RequestsWithTimeout)
    client.requests.get("url")
    assert seen == [("get", {"timeout": (2, 5)})]


def test_a_stalled_server_raises_instead_of_hanging():
    requests = pytest.importorskip("requests")
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    held = []
    threading.Thread(target=lambda: held.append(server.accept()), daemon=True).start()

    client = types.SimpleNamespace(__name__="stalled", requests=requests)
    http_timeouts.install(client, connect=2, read=1)
    started = time.monotonic()
    with pytest.raises(requests.exceptions.ReadTimeout):
        client.requests.get(f"http://127.0.0.1:{server.getsockname()[1]}/")
    assert time.monotonic() - started < 5
    server.close()
