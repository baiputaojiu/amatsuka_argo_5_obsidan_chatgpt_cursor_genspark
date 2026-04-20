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
        self.update_calls = []

    def patch(self, **kwargs):
        self.patch_calls.append(kwargs)
        event_id = kwargs.get("eventId", "patched-id")
        return _Req({"id": event_id})

    def insert(self, **kwargs):
        self.insert_calls.append(kwargs)
        return _Req({"id": "created-id"})

    def update(self, **kwargs):
        self.update_calls.append(kwargs)
        event_id = kwargs.get("eventId", "updated-id")
        return _Req({"id": event_id})


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


def _install_service() -> _Service:
    service = _Service()
    gc._SERVICE_LOCAL = threading.local()
    gc._SERVICE_LOCAL.service = service
    return service


def test_upsert_event_patch_when_no_existing_event():
    """既存イベント未指定のときは従来どおり patch を使うこと。"""
    service = _install_service()

    body = {"summary": "Test", "start": {"date": "2026-05-01"}, "end": {"date": "2026-05-02"}}
    action, eid = gc.upsert_event("primary", "event-1", body)

    assert action == "updated"
    assert eid == "event-1"
    assert len(service.events_api.patch_calls) == 1
    assert len(service.events_api.update_calls) == 0


def test_upsert_event_patch_when_type_matches():
    """既存が date、送信も date なら patch のまま。"""
    service = _install_service()

    existing = {
        "id": "event-1",
        "start": {"date": "2026-05-01"},
        "end": {"date": "2026-05-02"},
        "summary": "Old",
    }
    body = {
        "summary": "New",
        "start": {"date": "2026-05-01"},
        "end": {"date": "2026-05-02"},
    }
    action, eid = gc.upsert_event("primary", "event-1", body, existing_event=existing)

    assert action == "updated"
    assert eid == "event-1"
    assert len(service.events_api.patch_calls) == 1
    assert len(service.events_api.update_calls) == 0


def test_upsert_event_falls_back_to_update_when_type_mismatches():
    """既存が dateTime のとき date で送ると update() にフォールバックすること。"""
    service = _install_service()

    existing = {
        "id": "event-1",
        "etag": "etag-xyz",
        "kind": "calendar#event",
        "htmlLink": "https://example/e",
        "created": "2025-01-01T00:00:00Z",
        "updated": "2025-01-02T00:00:00Z",
        "summary": "Old",
        "description": "既存の説明",
        "location": "旧・場所",
        "start": {"dateTime": "2026-05-01T00:00:00+09:00", "timeZone": "Asia/Tokyo"},
        "end": {"dateTime": "2026-05-02T00:00:00+09:00", "timeZone": "Asia/Tokyo"},
        "extendedProperties": {
            "private": {
                "tool_marker": gc.TOOL_MARKER,
                "sync_key": "abc",
            },
            "shared": {"foo": "bar"},
        },
    }
    body = {
        "summary": "メーデー",
        "description": "新しい説明",
        "start": {"date": "2026-05-01"},
        "end": {"date": "2026-05-02"},
        "extendedProperties": {
            "private": {"sync_key": "abc", "last_tool_write_utc": "overwrite-me"},
        },
    }

    action, eid = gc.upsert_event("primary", "event-1", body, existing_event=existing)

    assert action == "updated"
    assert eid == "event-1"
    assert len(service.events_api.patch_calls) == 0
    assert len(service.events_api.update_calls) == 1

    update_kwargs = service.events_api.update_calls[0]
    assert update_kwargs["eventId"] == "event-1"
    sent = update_kwargs["body"]

    # 読み取り専用/派生フィールドは剥がれていること
    for key in ("etag", "kind", "id", "created", "updated", "htmlLink"):
        assert key not in sent

    # start/end は完全置換され、反対側のフィールドは残存しない
    assert sent["start"] == {"date": "2026-05-01"}
    assert sent["end"] == {"date": "2026-05-02"}

    # 送信側で上書きしたフィールドは送信値に、未指定の既存フィールドは保持される
    assert sent["summary"] == "メーデー"
    assert sent["description"] == "新しい説明"
    assert sent["location"] == "旧・場所"

    # extendedProperties は既存 private と新 private がマージされ、shared は保持される
    priv = sent["extendedProperties"]["private"]
    assert priv["sync_key"] == "abc"
    assert priv["tool_marker"] == gc.TOOL_MARKER
    assert priv["last_tool_write_utc"] != "overwrite-me"
    assert priv["last_tool_write_utc"].endswith("Z")
    assert sent["extendedProperties"]["shared"] == {"foo": "bar"}


def test_upsert_event_falls_back_when_existing_date_body_datetime():
    """逆方向（既存 date → 送信 dateTime）も update に切り替わること。"""
    service = _install_service()

    existing = {
        "id": "event-2",
        "start": {"date": "2026-05-01"},
        "end": {"date": "2026-05-02"},
    }
    body = {
        "summary": "メーデー",
        "start": {"dateTime": "2026-05-01T09:00:00", "timeZone": "Asia/Tokyo"},
        "end": {"dateTime": "2026-05-01T10:00:00", "timeZone": "Asia/Tokyo"},
    }

    action, eid = gc.upsert_event("primary", "event-2", body, existing_event=existing)

    assert action == "updated"
    assert eid == "event-2"
    assert len(service.events_api.update_calls) == 1
    assert len(service.events_api.patch_calls) == 0
    sent = service.events_api.update_calls[0]["body"]
    assert sent["start"] == {
        "dateTime": "2026-05-01T09:00:00",
        "timeZone": "Asia/Tokyo",
    }
    assert "date" not in sent["start"]


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
