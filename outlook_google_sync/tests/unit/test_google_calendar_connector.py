from __future__ import annotations

import threading

from outlook_google_sync.connectors import google_calendar as gc


class _Req:
    def __init__(self, result: dict):
        self._result = result

    def execute(self):
        return self._result


class _EventsApi:
    def __init__(self):
        self.patch_calls = []
        self.insert_calls = []

    def patch(self, **kwargs):
        self.patch_calls.append(kwargs)
        event_id = kwargs.get("eventId", "patched-id")
        return _Req({"id": event_id})

    def insert(self, **kwargs):
        self.insert_calls.append(kwargs)
        return _Req({"id": "created-id"})


class _Service:
    def __init__(self):
        self.events_api = _EventsApi()

    def events(self):
        return self.events_api


def test_upsert_event_single_write_and_stamp():
    service = _Service()
    gc._SERVICE_LOCAL = threading.local()
    gc._SERVICE_LOCAL.service = service

    body = {"summary": "Test"}
    action, eid = gc.upsert_event("primary", "event-1", body)

    assert action == "updated"
    assert eid == "event-1"
    assert len(service.events_api.patch_calls) == 1
    call_body = service.events_api.patch_calls[0]["body"]
    private = (call_body.get("extendedProperties") or {}).get("private") or {}
    assert private.get("tool_marker") == gc.TOOL_MARKER
    assert isinstance(private.get("last_tool_write_utc"), str)
    assert private["last_tool_write_utc"].endswith("Z")


def test_get_service_reuses_thread_local_service(monkeypatch, tmp_path):
    build_calls = {"count": 0}
    built_service = object()

    class _Creds:
        valid = True
        expired = False
        refresh_token = None

        def to_json(self):
            return "{}"

    class _Flow:
        def run_local_server(self, port=0):
            return _Creds()

    def fake_build(_api, _ver, credentials=None):
        build_calls["count"] += 1
        return built_service

    monkeypatch.setattr(gc, "_SERVICE_LOCAL", threading.local())
    monkeypatch.setattr(gc, "token_path", lambda: tmp_path / "token.json")
    monkeypatch.setattr(gc, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.setattr(gc.InstalledAppFlow, "from_client_secrets_file", lambda *a, **k: _Flow())
    monkeypatch.setattr(gc, "build", fake_build)

    s1 = gc.get_service()
    s2 = gc.get_service()

    assert s1 is built_service
    assert s2 is built_service
    assert build_calls["count"] == 1
