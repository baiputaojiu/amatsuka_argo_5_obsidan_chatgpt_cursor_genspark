"""Phase 3 tests: pagination, retry, and get_service locking.

These cover behaviours that were either missing (pagination) or
previously implicit (retry + init lock) in the Google Calendar connector.
"""

from __future__ import annotations

import threading
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError

from outlook_google_sync.connectors import google_calendar as gc


# ────────────────────────────────────────────────────────────────────
# Helpers: minimal fakes for the googleapiclient service surface
# ────────────────────────────────────────────────────────────────────


class _FakeResp:
    """Mimic ``httplib2.Response``.

    ``googleapiclient.errors.HttpError`` reads ``status`` + ``reason`` on
    construction and exposes ``get`` for header lookups; we provide the
    minimum surface to keep tests decoupled from the real HTTP layer.
    """

    def __init__(self, status: int, headers: dict | None = None):
        self.status = status
        self.reason = "simulated"
        self._headers = {k.lower(): v for k, v in (headers or {}).items()}

    def get(self, key, default=None):
        return self._headers.get(key.lower(), default)


def _make_http_error(status: int, headers: dict | None = None) -> HttpError:
    resp = _FakeResp(status, headers)
    return HttpError(resp=resp, content=b"{}")


class _ScriptedRequest:
    """Request object whose ``execute()`` returns results from a script.

    Each element of ``script`` is either a value (returned) or an
    exception (raised). After the list is exhausted, subsequent calls
    repeat the last entry so tests don't accidentally IndexError on
    retries that happen to succeed.
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.calls = 0

    def execute(self):
        self.calls += 1
        entry = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(entry, Exception):
            raise entry
        return entry


# ────────────────────────────────────────────────────────────────────
# _execute_with_retry
# ────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Never actually sleep in tests — capture calls instead."""
    calls: list[float] = []
    monkeypatch.setattr(gc, "_sleep", lambda s: calls.append(s))
    return calls


def test_retry_succeeds_after_transient_500(_no_real_sleep):
    req = _ScriptedRequest([_make_http_error(500), {"ok": True}])
    result = gc._execute_with_retry(req, label="test")
    assert result == {"ok": True}
    assert req.calls == 2
    assert len(_no_real_sleep) == 1  # slept once before retry


def test_retry_honours_retry_after_on_429(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(gc, "_sleep", lambda s: slept.append(s))
    req = _ScriptedRequest([
        _make_http_error(429, headers={"Retry-After": "7"}),
        {"ok": True},
    ])
    gc._execute_with_retry(req, label="test")
    assert slept == [7.0]


def test_retry_uses_computed_backoff_when_no_retry_after(monkeypatch):
    # Force deterministic jitter
    monkeypatch.setattr(gc.random, "uniform", lambda a, b: b)
    slept: list[float] = []
    monkeypatch.setattr(gc, "_sleep", lambda s: slept.append(s))
    req = _ScriptedRequest([
        _make_http_error(503),
        _make_http_error(503),
        {"ok": True},
    ])
    gc._execute_with_retry(req, label="test")
    # Two retries → two sleeps; first uses base, second uses base * 2
    assert slept == [gc.GOOGLE_API_RETRY_BASE_DELAY_SECONDS,
                     gc.GOOGLE_API_RETRY_BASE_DELAY_SECONDS * 2]


def test_retry_exhausts_and_raises():
    err = _make_http_error(503)
    req = _ScriptedRequest([err] * 10)
    with pytest.raises(HttpError):
        gc._execute_with_retry(req, label="test")
    assert req.calls == gc.GOOGLE_API_MAX_RETRY_ATTEMPTS


def test_non_retryable_status_propagates_immediately():
    req = _ScriptedRequest([_make_http_error(404)])
    with pytest.raises(HttpError):
        gc._execute_with_retry(req, label="test")
    assert req.calls == 1


def test_non_http_exception_propagates():
    class _CustomErr(Exception):
        pass

    req = _ScriptedRequest([_CustomErr("boom")])
    with pytest.raises(_CustomErr):
        gc._execute_with_retry(req, label="test")
    assert req.calls == 1


def test_backoff_is_capped():
    # Attempt index high enough to exceed cap before jitter
    delay = gc._compute_backoff(attempt=20)
    assert 0.0 <= delay <= gc.GOOGLE_API_RETRY_MAX_DELAY_SECONDS


# ────────────────────────────────────────────────────────────────────
# _paginate_events_list
# ────────────────────────────────────────────────────────────────────


class _EventsListStub:
    """Fake ``service.events()`` supporting only ``.list(...).execute()``.

    Returns pages in order. Each call to ``.list(...)`` records kwargs so
    tests can assert on ``pageToken`` propagation.
    """

    def __init__(self, pages: list[dict]):
        self.pages = pages
        self.list_calls: list[dict] = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        idx = len(self.list_calls) - 1
        resp = self.pages[idx] if idx < len(self.pages) else {}
        return _ScriptedRequest([resp])


class _ServiceStub:
    def __init__(self, events_stub: _EventsListStub):
        self._events = events_stub

    def events(self):
        return self._events


def _install_service(monkeypatch, events_stub: _EventsListStub) -> _ServiceStub:
    svc = _ServiceStub(events_stub)
    monkeypatch.setattr(gc, "get_service", lambda: svc)
    return svc


def test_pagination_drains_all_pages(monkeypatch):
    pages = [
        {"items": [{"id": "a"}, {"id": "b"}], "nextPageToken": "tok1"},
        {"items": [{"id": "c"}], "nextPageToken": "tok2"},
        {"items": [{"id": "d"}]},  # last page: no nextPageToken
    ]
    events_stub = _EventsListStub(pages)
    _install_service(monkeypatch, events_stub)

    out = list(gc._paginate_events_list(
        "primary",
        datetime(2026, 1, 1),
        datetime(2026, 2, 1),
        label="test",
    ))
    ids = [e["id"] for e in out]
    assert ids == ["a", "b", "c", "d"]
    # First call has no pageToken; subsequent carry the prior token
    assert events_stub.list_calls[0].get("pageToken") is None
    assert events_stub.list_calls[1]["pageToken"] == "tok1"
    assert events_stub.list_calls[2]["pageToken"] == "tok2"


def test_pagination_terminates_on_empty_page(monkeypatch):
    pages = [{"items": []}]
    events_stub = _EventsListStub(pages)
    _install_service(monkeypatch, events_stub)

    out = list(gc._paginate_events_list(
        "primary",
        datetime(2026, 1, 1),
        datetime(2026, 2, 1),
        label="test",
    ))
    assert out == []
    assert len(events_stub.list_calls) == 1


def test_list_managed_events_paginates_and_filters(monkeypatch):
    managed_a = {
        "id": "a",
        "extendedProperties": {"private": {
            "tool_marker": gc.TOOL_MARKER,
            "sync_key": "k1",
        }},
    }
    unmanaged = {"id": "b"}
    managed_c = {
        "id": "c",
        "extendedProperties": {"private": {
            "tool_marker": gc.TOOL_MARKER,
            "sync_key": "k2",
        }},
    }
    pages = [
        {"items": [managed_a, unmanaged], "nextPageToken": "t1"},
        {"items": [managed_c]},
    ]
    _install_service(monkeypatch, _EventsListStub(pages))

    out = gc.list_managed_events("primary", datetime(2026, 1, 1), datetime(2026, 2, 1))
    assert set(out.keys()) == {"k1", "k2"}
    assert out["k1"]["id"] == "a"
    assert out["k2"]["id"] == "c"


def test_list_all_events_in_range_returns_everything(monkeypatch):
    pages = [
        {"items": [{"id": "a"}, {"id": "b"}], "nextPageToken": "t1"},
        {"items": [{"id": "c"}]},
    ]
    _install_service(monkeypatch, _EventsListStub(pages))
    out = gc.list_all_events_in_range("primary", datetime(2026, 1, 1), datetime(2026, 2, 1))
    assert [e["id"] for e in out] == ["a", "b", "c"]


# ────────────────────────────────────────────────────────────────────
# get_service init lock
# ────────────────────────────────────────────────────────────────────


def test_get_service_concurrent_cold_start_runs_auth_once(monkeypatch, tmp_path):
    """Multiple cold-start threads must share a single OAuth flow run.

    After the first thread writes ``token.json``, the other threads hit
    the ``tp.exists()`` branch and load the cached creds instead of
    re-running the OAuth flow.
    """

    oauth_flow_calls: list[int] = []
    oauth_flow_lock = threading.Lock()

    class _Creds:
        valid = True
        expired = False
        refresh_token = None

        def to_json(self):
            return "{}"

    class _Flow:
        def run_local_server(self, port=0):
            with oauth_flow_lock:
                oauth_flow_calls.append(1)
            # Simulate slow OAuth so all threads queue on _CREDS_LOCK.
            import time as _t
            _t.sleep(0.05)
            return _Creds()

    build_calls: list[int] = []

    def fake_build(_api, _ver, credentials=None):
        build_calls.append(1)
        return MagicMock(name=f"service-{len(build_calls)}")

    # Fake Credentials.from_authorized_user_file so non-first threads can
    # "load" the written token.json (which, in these tests, contains only
    # a placeholder blob).
    fake_creds_cls = MagicMock()
    fake_creds_cls.from_authorized_user_file = MagicMock(return_value=_Creds())

    fresh_local = threading.local()
    monkeypatch.setattr(gc, "_SERVICE_LOCAL", fresh_local)
    monkeypatch.setattr(gc, "_CREDS_LOCK", threading.Lock())
    monkeypatch.setattr(gc, "token_path", lambda: tmp_path / "token.json")
    monkeypatch.setattr(gc, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.setattr(gc, "Credentials", fake_creds_cls)
    monkeypatch.setattr(gc.InstalledAppFlow, "from_client_secrets_file", lambda *a, **k: _Flow())
    monkeypatch.setattr(gc, "build", fake_build)

    results: list[object] = []

    def worker():
        results.append(gc.get_service())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each thread built its own client (thread-local cache). The OAuth
    # flow — protected by _CREDS_LOCK — ran exactly once because the first
    # thread wrote token.json before the others reached the load step.
    assert oauth_flow_calls == [1]
    assert len(build_calls) == 4
    assert len(results) == 4


def test_get_service_returns_cached_service_in_same_thread(monkeypatch, tmp_path):
    """Second call on the same thread must skip auth entirely."""

    class _Creds:
        valid = True
        expired = False
        refresh_token = None

        def to_json(self):
            return "{}"

    class _Flow:
        def run_local_server(self, port=0):
            return _Creds()

    build_calls: list[int] = []

    def fake_build(_api, _ver, credentials=None):
        build_calls.append(1)
        return object()

    monkeypatch.setattr(gc, "_SERVICE_LOCAL", threading.local())
    monkeypatch.setattr(gc, "_CREDS_LOCK", threading.Lock())
    monkeypatch.setattr(gc, "token_path", lambda: tmp_path / "token.json")
    monkeypatch.setattr(gc, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.setattr(gc.InstalledAppFlow, "from_client_secrets_file", lambda *a, **k: _Flow())
    monkeypatch.setattr(gc, "build", fake_build)

    s1 = gc.get_service()
    s2 = gc.get_service()
    assert s1 is s2
    assert len(build_calls) == 1
