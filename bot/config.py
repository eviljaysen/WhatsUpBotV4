"""bot/config.py — Configuration, constants, and OCR correction persistence.

Single source of truth for all settings. All other modules import constants
from here. Config never imports from other bot/ modules.

Supports hot-reload: call reload_config() at the start of each scan to pick
up changes to config.json (new players, corrections, coords) without restart.
"""

import os
import json

# ── Paths ──────────────────────────────────────────────────────────────────────
import sys

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGES_DIR        = os.path.join(BASE_DIR, "images")
BUILDS_DIR        = os.path.join(BASE_DIR, "builds")
BUILD_HISTORY_DIR = os.path.join(BASE_DIR, "build_history")
SCANS_DIR         = os.path.join(BASE_DIR, "scans")
NAME_TMPL_DIR     = os.path.join(BASE_DIR, "name_templates")
DB_PATH           = os.path.join(BASE_DIR, "history.db")

for _d in (IMAGES_DIR, BUILDS_DIR, BUILD_HISTORY_DIR, SCANS_DIR, NAME_TMPL_DIR):
    os.makedirs(_d, exist_ok=True)

_CFG_PATH = os.path.join(BASE_DIR, "config.json")
_cfg_mtime: float = 0.0   # last known mtime of config.json


# ── Config load ────────────────────────────────────────────────────────────────
def _load_raw() -> dict:
    try:
        with open(_CFG_PATH, encoding="utf-8") as _f:
            return json.load(_f)
    except FileNotFoundError:
        print(f"[config] config.json not found at {_CFG_PATH} — using defaults")
        return {}
    except json.JSONDecodeError as e:
        print(f"[config] JSON parse error: {e} — using defaults")
        return {}


CFG: dict = _load_raw()

# Mutable containers — initialised here, mutated in-place by _derive() so that
# `from bot.config import player_timezones` refs stay valid after reload.
player_timezones: dict = {}
_OCR_CORRECTIONS: dict = {}
_KNOWN_NAMES:     list = []
_NAME_UPPER_MAP:  dict = {}
_ENEMY_BLDGS_CFG: dict = {}


def _derive(cfg: dict):
    """Derive all module-level constants from a config dict.

    Mutable containers are mutated IN-PLACE so that references held by other
    modules (via ``from bot.config import X``) stay valid after reload.
    Scalars are reassigned — other modules should read them from CFG.get()
    if they need the post-reload value.
    """
    global GAME_WINDOW_TITLE, CONTENT_OFFSET_X, CONTENT_OFFSET_Y
    global NAME_MATCH_CUTOFF, BOT_SLOTS_TOTAL
    global SLEEP_START, SLEEP_END, _CJK_NAMES
    global _TMPL_THRESHOLD, _TMPL_CONFIDENCE, _CJK_MATCH_CUTOFF
    global DISCORD_WEBHOOK_URL, SCAN_INTERVAL_MINUTES
    global EASYOCR_ENABLED, EASYOCR_GPU

    # Scalars — reassigned (use CFG.get() in other modules for live value)
    GAME_WINDOW_TITLE    = cfg.get("game_window_title",     "BlueStacks")
    CONTENT_OFFSET_X     = cfg.get("content_offset_x",     0)
    CONTENT_OFFSET_Y     = cfg.get("content_offset_y",     0)
    SLEEP_START          = cfg.get("sleep_start",           21)
    SLEEP_END            = cfg.get("sleep_end",             8)
    NAME_MATCH_CUTOFF    = cfg.get("name_match_cutoff",     0.80)
    BOT_SLOTS_TOTAL      = cfg.get("bot_slots_total",       75)
    _CJK_NAMES           = cfg.get("cjk_names",             False)
    _TMPL_THRESHOLD      = cfg.get("tmpl_threshold",        0.40)
    _TMPL_CONFIDENCE     = cfg.get("tmpl_confidence",       0.75)
    _CJK_MATCH_CUTOFF    = cfg.get("cjk_match_cutoff",      0.65)
    DISCORD_WEBHOOK_URL  = cfg.get("discord_webhook_url",   "")
    SCAN_INTERVAL_MINUTES= cfg.get("scan_interval_minutes", 0)
    EASYOCR_ENABLED      = cfg.get("easyocr_enabled",       True)
    EASYOCR_GPU          = cfg.get("easyocr_gpu",            False)

    # Mutable containers — mutate in-place so imported refs stay valid
    player_timezones.clear()
    player_timezones.update(cfg.get("players", {}))

    _OCR_CORRECTIONS.clear()
    _OCR_CORRECTIONS.update(cfg.get("ocr_corrections", {}))

    _ENEMY_BLDGS_CFG.clear()
    _ENEMY_BLDGS_CFG.update(cfg.get("enemy_buildings", {}))

    _KNOWN_NAMES.clear()
    _KNOWN_NAMES.extend(player_timezones.keys())

    _NAME_UPPER_MAP.clear()
    _NAME_UPPER_MAP.update({n.upper(): n for n in _KNOWN_NAMES})


_derive(CFG)
try:
    _cfg_mtime = os.path.getmtime(_CFG_PATH)
except OSError:
    pass


def reload_config() -> dict:
    """Reload config.json and update all derived constants. Call at scan start.

    Skips reload if config.json hasn't been modified since last load (mtime check).
    """
    global _cfg_mtime
    try:
        current_mtime = os.path.getmtime(_CFG_PATH)
    except OSError:
        current_mtime = 0.0

    if current_mtime == _cfg_mtime and _cfg_mtime > 0:
        print("[config] config.json unchanged — skipping reload")
        return CFG

    new = _load_raw()
    CFG.clear()
    CFG.update(new)
    _derive(CFG)
    _cfg_mtime = current_mtime
    # Re-apply HUD overrides from updated config (late import avoids circular dep)
    from bot.templates import reload_hud_overrides
    reload_hud_overrides()
    print("[config] Reloaded config.json")
    return CFG


# ── OCR correction persistence ─────────────────────────────────────────────────
def save_hud_overrides(overrides: dict):
    """Persist HUD region overrides to config.json under 'hud_overrides'.

    Args:
        overrides: dict of region_name → (bx, by, bw, bh) in 1920×1080 baseline.
    """
    try:
        with open(_CFG_PATH, 'r', encoding='utf-8') as _f:
            cfg = json.load(_f)
        # Store as lists for JSON serialization
        cfg["hud_overrides"] = {k: list(v) for k, v in overrides.items()}
        with open(_CFG_PATH, 'w', encoding='utf-8') as _f:
            json.dump(cfg, _f, indent=4, ensure_ascii=False)
        print(f"[config] Saved HUD overrides: {list(overrides.keys())}")
    except Exception as e:
        print(f"[config] Failed to save HUD overrides: {e}")


def save_ocr_correction(raw_upper: str, corrected: str):
    """Persist a raw→corrected OCR mapping to _OCR_CORRECTIONS and config.json."""
    if len(raw_upper.strip()) < 2:
        print(f"[config] Skipped correction: key too short ({raw_upper!r})")
        return
    _OCR_CORRECTIONS[raw_upper] = corrected
    try:
        with open(_CFG_PATH, 'r', encoding='utf-8') as _f:
            cfg = json.load(_f)
        cfg.setdefault("ocr_corrections", {})[raw_upper] = corrected
        with open(_CFG_PATH, 'w', encoding='utf-8') as _f:
            json.dump(cfg, _f, indent=4, ensure_ascii=False)
        print(f"[config] Saved correction: {raw_upper!r} → {corrected!r}")
    except Exception as e:
        print(f"[config] Failed to save correction: {e}")
