"""Unit tests: services.setup_checklist."""

from outlook_google_sync.services import setup_checklist as sc


def test_both_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(sc, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.setattr(sc, "token_path", lambda: tmp_path / "token.json")
    items = {i.id: i for i in sc.evaluate_setup({"calendar_id": "primary"})}
    assert items["credentials"].done is False
    assert items["token"].done is False
    assert items["calendar"].done is True
    assert sc.is_setup_incomplete({"calendar_id": "primary"}) is True


def test_credentials_only(monkeypatch, tmp_path):
    creds = tmp_path / "credentials.json"
    creds.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sc, "credentials_path", lambda: creds)
    monkeypatch.setattr(sc, "token_path", lambda: tmp_path / "token.json")
    assert sc.is_setup_incomplete({"calendar_id": "primary"}) is True
    items = {i.id: i for i in sc.evaluate_setup({"calendar_id": "primary"})}
    assert items["credentials"].done is True
    assert items["token"].done is False


def test_both_present(monkeypatch, tmp_path):
    creds = tmp_path / "credentials.json"
    token = tmp_path / "token.json"
    creds.write_text("{}", encoding="utf-8")
    token.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sc, "credentials_path", lambda: creds)
    monkeypatch.setattr(sc, "token_path", lambda: token)
    assert sc.is_setup_incomplete({"calendar_id": "primary"}) is False
    items = {i.id: i for i in sc.evaluate_setup({"calendar_id": "primary"})}
    assert items["credentials"].done is True
    assert items["token"].done is True


def test_empty_calendar_id(monkeypatch, tmp_path):
    creds = tmp_path / "credentials.json"
    token = tmp_path / "token.json"
    creds.write_text("{}", encoding="utf-8")
    token.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sc, "credentials_path", lambda: creds)
    monkeypatch.setattr(sc, "token_path", lambda: token)
    items = {i.id: i for i in sc.evaluate_setup({"calendar_id": ""})}
    assert items["calendar"].done is False
    # calendar alone must not force the dialog when creds+token exist
    assert sc.is_setup_incomplete({"calendar_id": ""}) is False
