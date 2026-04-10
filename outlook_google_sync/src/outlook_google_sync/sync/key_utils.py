from ..utils.hash_utils import sha256_text

def generate_fallback_sync_key(reader_engine:str, summary:str, start_iso:str, end_iso:str)->str:
    return sha256_text(f"{reader_engine}|{summary}|{start_iso}|{end_iso}")[:32]
