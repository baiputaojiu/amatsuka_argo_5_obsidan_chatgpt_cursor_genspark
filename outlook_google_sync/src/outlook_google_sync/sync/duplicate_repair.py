from collections import defaultdict

def find_duplicates(events):
    groups = defaultdict(list)
    for item in events:
        private = ((item.get("extendedProperties") or {}).get("private") or {})
        key = private.get("sync_key")
        if key: groups[key].append(item)
    return {k:v for k,v in groups.items() if len(v)>1}
