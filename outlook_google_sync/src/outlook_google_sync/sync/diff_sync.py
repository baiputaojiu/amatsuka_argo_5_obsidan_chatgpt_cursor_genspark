def filter_diff_targets(events, previous_fingerprints: dict, google_existing: dict):
    targets = []
    for e in events:
        prev = previous_fingerprints.get(e.sync_key)
        if prev != e.fingerprint:
            targets.append(e); continue
        item = google_existing.get(e.sync_key)
        if not item:
            targets.append(e); continue
        private = ((item.get("extendedProperties") or {}).get("private") or {})
        last_write = private.get("last_tool_write_utc")
        updated = item.get("updated")
        if not last_write or not updated or updated > last_write:
            targets.append(e)
    return targets
