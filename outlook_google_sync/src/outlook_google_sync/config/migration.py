import json, shutil
from .paths import config_path
from ..models.config_schema import SCHEMA_VERSION

def migrate_config(path):
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "runtime_state": {}, "sync_metadata": {"per_source_fingerprint": {}}}
    data = json.loads(path.read_text(encoding="utf-8"))
    old = int(data.get("schema_version", 0))
    if old == SCHEMA_VERSION:
        return data
    if old < SCHEMA_VERSION:
        data.setdefault("runtime_state", {})
        rs = data["runtime_state"]
        rs.setdefault("duplicate_repair_mode", "sync_key")
        rs.setdefault("duplicate_repair_description_mode", "longer")
        data.setdefault("sync_metadata", {"per_source_fingerprint": {}})
        data["schema_version"] = SCHEMA_VERSION
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data
    backup = path.with_name(f"config.backup.v{old}.json")
    shutil.copy2(path, backup)
    raise ValueError(f"Unsupported schema version: {old}. backup={backup}")
