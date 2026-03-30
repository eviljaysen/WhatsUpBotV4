"""bot/config.py — Configuration, constants, and OCR correction persistence.

Single source of truth for all settings. All other modules import constants
from here. Config never imports from other bot/ modules.

Supports hot-reload: call reload_config() at the start of each scan to pick
up changes to config.json (new players, corrections, coords) without restart.
"""

import os
import json
import logging
from logging.handlers import RotatingFileHandler

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
LOG_PATH          = os.path.join(BASE_DIR, "bot.log")

for _d in (IMAGES_DIR, BUILDS_DIR, BUILD_HISTORY_DIR, SCANS_DIR, NAME_TMPL_DIR):
    os.makedirs(_d, exist_ok=True)


# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a named logger. All bot loggers share the 'bot' root handler."""
    return logging.getLogger(f"bot.{name}")


def _init_logging():
    """Set up root 'bot' logger with rotating file + console output."""
    root = logging.getLogger("bot")
    if root.handlers:
        return  # already initialised
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler: 2 MB max, keep 3 backups
    fh = RotatingFileHandler(
        LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler (INFO and above — keeps terminal output similar to before)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    root.addHandler(ch)


_init_logging()
_log = get_logger("config")


def truncate_log():
    """Clear the log file so each scan starts fresh.

    Flushes and truncates the rotating file handler's current file
    without removing backup files.
    """
    root = logging.getLogger("bot")
    for handler in root.handlers:
        if isinstance(handler, RotatingFileHandler):
            handler.flush()
            with open(handler.baseFilename, "w", encoding="utf-8"):
                pass  # truncate to zero bytes
            break

_CFG_PATH = os.path.join(BASE_DIR, "config.json")
_cfg_mtime: float = 0.0   # last known mtime of config.json


# ── Config load ────────────────────────────────────────────────────────────────
def _load_raw() -> dict:
    try:
        with open(_CFG_PATH, encoding="utf-8") as _f:
            return json.load(_f)
    except FileNotFoundError:
        _log.warning("config.json not found at %s — using defaults", _CFG_PATH)
        return {}
    except json.JSONDecodeError as e:
        _log.error("JSON parse error: %s — using defaults", e)
        return {}


CFG: dict = _load_raw()

# Mutable containers — initialised here, mutated in-place by _derive() so that
# `from bot.config import player_timezones` refs stay valid after reload.
player_timezones: dict = {}
_OCR_CORRECTIONS: dict = {}
_KNOWN_NAMES:     list = []
_NAME_UPPER_MAP:  dict = {}
_ENEMY_BLDGS_CFG: dict = {}

# OCR sanity bounds — set by _derive(), importable by other modules
MAX_HP_PER_CAR:   int = 8_000_000
MAX_ATK_PER_CAR:  int = 8_000_000
MAX_PLAYER_TOTAL: int = 16_000_000
MIN_STAT_VALUE:   int = 1000


def _derive(cfg: dict):
    """Derive all module-level constants from a config dict.

    Mutable containers are mutated IN-PLACE so that references held by other
    modules (via ``from bot.config import X``) stay valid after reload.
    Scalars are reassigned — other modules should read them from CFG.get()
    if they need the post-reload value.
    """
    global GAME_WINDOW_TITLE, CONTENT_OFFSET_X, CONTENT_OFFSET_Y
    global NAME_MATCH_CUTOFF, BOT_SLOTS_TOTAL
    global SLEEP_START, SLEEP_END
    global DISCORD_WEBHOOK_URL, SCAN_INTERVAL_MINUTES

    # Scalars — reassigned (use CFG.get() in other modules for live value)
    GAME_WINDOW_TITLE    = cfg.get("game_window_title",     "BlueStacks")
    CONTENT_OFFSET_X     = cfg.get("content_offset_x",     0)
    CONTENT_OFFSET_Y     = cfg.get("content_offset_y",     0)
    SLEEP_START          = cfg.get("sleep_start",           21)
    SLEEP_END            = cfg.get("sleep_end",             8)
    NAME_MATCH_CUTOFF    = cfg.get("name_match_cutoff",     0.80)
    BOT_SLOTS_TOTAL      = cfg.get("bot_slots_total",       75)
    DISCORD_WEBHOOK_URL  = cfg.get("discord_webhook_url",   "")
    SCAN_INTERVAL_MINUTES= cfg.get("scan_interval_minutes", 0)

    # OCR sanity bounds — values outside these ranges are rejected as garbage.
    # Per-car: no single car should exceed these values.
    # Per-player: total across all cars (max 3) should not exceed these.
    global MAX_HP_PER_CAR, MAX_ATK_PER_CAR, MAX_PLAYER_TOTAL, MIN_STAT_VALUE
    MAX_HP_PER_CAR     = cfg.get("max_hp_per_car",      8_000_000)
    MAX_ATK_PER_CAR    = cfg.get("max_atk_per_car",     8_000_000)
    MAX_PLAYER_TOTAL   = cfg.get("max_player_total",   16_000_000)
    MIN_STAT_VALUE     = cfg.get("min_stat_value",           1000)

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
        _log.debug("config.json unchanged — skipping reload")
        return CFG

    new = _load_raw()
    CFG.clear()
    CFG.update(new)
    _derive(CFG)
    _cfg_mtime = current_mtime
    # Re-apply HUD overrides from updated config (late import avoids circular dep)
    from bot.templates import reload_hud_overrides
    reload_hud_overrides()
    _log.info("Reloaded config.json")
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
        _log.info("Saved HUD overrides: %s", list(overrides.keys()))
    except Exception as e:
        _log.error("Failed to save HUD overrides: %s", e)


def save_ocr_correction(raw_upper: str, corrected: str):
    """Persist a raw→corrected OCR mapping to _OCR_CORRECTIONS and config.json."""
    if len(raw_upper.strip()) < 2:
        _log.warning("Skipped correction: key too short (%r)", raw_upper)
        return
    _OCR_CORRECTIONS[raw_upper] = corrected
    try:
        with open(_CFG_PATH, 'r', encoding='utf-8') as _f:
            cfg = json.load(_f)
        cfg.setdefault("ocr_corrections", {})[raw_upper] = corrected
        with open(_CFG_PATH, 'w', encoding='utf-8') as _f:
            json.dump(cfg, _f, indent=4, ensure_ascii=False)
        _log.info("Saved correction: %r → %r", raw_upper, corrected)
    except Exception as e:
        _log.error("Failed to save correction: %s", e)
