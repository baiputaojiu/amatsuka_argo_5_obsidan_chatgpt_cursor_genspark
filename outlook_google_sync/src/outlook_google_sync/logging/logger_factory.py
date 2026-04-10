import logging
from datetime import datetime
from ..config.paths import logs_dir

def build_logger(name="outlook_google_sync"):
    logs_dir().mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers: return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(logs_dir()/f"sync_{datetime.now().date()}.log", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger
