import json
import os
from threading import Lock

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
_lock = Lock()
_config_cache = None

def _load():
    global _config_cache
    with _lock:
        if _config_cache is None:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _config_cache = json.load(f)
        return _config_cache

def reload_config():
    global _config_cache
    with _lock:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config_cache = json.load(f)
    return _config_cache

def get(key=None, default=None):
    cfg = _load()
    if not key:
        return cfg
    parts = key.split(".")
    cur = cfg
    try:
        for p in parts:
            cur = cur[p]
        return cur
    except (KeyError, TypeError):
        return default

def set(key, value):
    cfg = _load()
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            raise KeyError(f"Key path invalid: {'.'.join(parts[:-1])}")
        cur = cur[p]
    last = parts[-1]
    if last not in cur:
        raise KeyError(f"Key '{key}' does not exist.")
    # optional: validate type here
    cur[last] = value
    # persist
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return True