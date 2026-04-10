from pathlib import Path

def runtime_dir()->Path: return Path.home()/".outlook_google_sync"
def config_path()->Path: return runtime_dir()/"config.json"
def token_path()->Path: return runtime_dir()/"token.json"
def credentials_path()->Path: return runtime_dir()/"credentials.json"
def logs_dir()->Path: return runtime_dir()/"logs"
