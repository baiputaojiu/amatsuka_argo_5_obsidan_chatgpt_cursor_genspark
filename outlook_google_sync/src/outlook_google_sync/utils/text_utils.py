def simple_partial_match(text:str, keywords:list[str])->bool:
    t=(text or "").lower().strip()
    return any(k.strip().lower() in t for k in keywords if k.strip())
