import json
from .paths import runtime_dir, config_path
from .migration import migrate_config

def load_settings()->dict:
    runtime_dir().mkdir(parents=True, exist_ok=True)
    return migrate_config(config_path())

def save_settings(data:dict)->None:
    runtime_dir().mkdir(parents=True, exist_ok=True)
    config_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
