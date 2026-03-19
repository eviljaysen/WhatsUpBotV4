# WhatsUpBot v5.0 — Claude Code & AI Agent Guide

> This file is loaded automatically by Claude Code. Read it before every task.
> It contains the full project contract: architecture, conventions, invariants, and anti-patterns.

---

## Project Purpose

WhatsUpBot scans a BlueStacks game window via screenshot + OCR to generate Discord defense
reports for a CK Alliance. It identifies players by name, captures build screenshots, reads
HP/ATK/score/timer values, and produces a formatted report with player stats, win projection,
bot availability, building strength analysis, placement recommendations, score momentum,
and configurable alerts.

---

## Quick Start

```bash
# Install deps
pip install -r requirements.txt

# Run
python WhatsUpBot.py

# Run tests
python -m pytest tests/ -v
```

**Config:** `config.json` — players, timezones, OCR corrections, building coords.
**Templates:** `images/` — required PNG templates for template matching.
**Builds:** `builds/` — current scan screenshots, copied to `build_history/` as stable 3-car record.
**Venv:** Always use the project venv. Never use system Python for this project.

---

## Architecture

```
WhatsUpBot.py            ← tkinter App UI + entry point ONLY
└── bot/
    ├── config.py        ← all constants, config load/reload, OCR correction persistence
    ├── window.py        ← Window class + BlueStacks window detection
    ├── templates.py     ← Templates cache (auto-rescaling), HUD coord map
    ├── vision.py        ← image converters, badge extraction, blob detection
    ├── ocr.py           ← all OCR: scores, timer, player names (ASCII + CJK), NameMatcher
    ├── name_model.py    ← CNN classifier for player name recognition (v5.0)
    ├── ocr_model.py     ← Character-level CNN for game OCR (v5.0)
    ├── easyocr_engine.py← Lazy-loaded EasyOCR wrapper (v4.1)
    ├── navigation.py    ← mouse/keyboard nav, slot cycling, building entry/exit
    ├── scan.py          ← ScanContext, run_team_scan, run_enemy_scan, auto-training
    ├── report.py        ← report builder, stats table, availability list, formatters
    ├── history.py       ← SQLite scan history + war tracking (v5.0)
    ├── analysis.py      ← Strategic analysis: buildings, momentum, recommendations (v5.0)
    ├── alerts.py        ← Alert evaluation + Discord firing (v5.0)
    └── discord_post.py  ← Discord webhook posting (v4.0)
```

**Threading model:**

- Main thread: tkinter event loop only. Never block it.
- Scan threads: `threading.Thread(daemon=True)` — one at a time, started by App.
- ML training thread: `threading.Thread(daemon=True)` — kicked off after each scan via `_auto_train_all()`.
- All tkinter mutations from threads: `root.after(0, lambda m=msg: fn(m))` — value-captured lambdas.

---

## Module Dependency Graph

```
WhatsUpBot.py
    └── bot.scan ──────────────────────────────────────────────┐
         ├── bot.navigation ──────────────────────────────────┐ │
         │    ├── bot.window                                   │ │
         │    ├── bot.templates                                │ │
         │    └── bot.config                                   │ │
         ├── bot.ocr ──────────────────────────────────────────│─┤
         │    ├── bot.vision                                   │ │
         │    ├── bot.easyocr_engine (lazy)                    │ │
         │    └── bot.config                                   │ │
         ├── bot.name_model (lazy, v5.0) ──────────────────────│─┤
         │    └── bot.config                                   │ │
         ├── bot.ocr_model  (lazy, v5.0) ──────────────────────│─┤
         │    └── bot.config                                   │ │
         ├── bot.report ─────────────────────────────────────── │
         │    └── bot.config                                     │
         ├── bot.analysis ─────────────────────────────────────── │
         │    └── bot.report                                     │
         ├── bot.alerts ──────────────────────────────────────── │
         │    ├── bot.analysis                                   │
         │    └── bot.discord_post                               │
         ├── bot.history (v5.0)                                  │
         └── bot.discord_post (v4.0)                            │
                                                                │
    bot.navigation also uses: bot.ocr, bot.vision              ┘
```

**Rule:** No circular imports. Lower modules never import from higher ones.
Import order: `config` → `window` → `templates` → `vision` → `easyocr_engine` → `ocr` → `name_model` → `ocr_model` → `navigation` → `analysis` → `alerts` → `scan` → `report`

**Note:** `name_model`, `ocr_model`, `easyocr_engine`, `history`, and `alerts` are imported lazily
(inside functions) from `scan.py` and `report.py` to avoid heavy startup costs (PyTorch, SQLite).

---

## The Central Data Structure: ScanContext

Every scan creates a `ScanContext` instance and passes it through the entire call chain.
**No module-level mutable state for scan data.** This replaces v3's `_logo_cache[0]`,
`_last_name_shot[0]`, `_correction_result[0]`, etc.

```python
@dataclasses.dataclass
class ScanContext:
    win:               Window
    cfg:               dict            # live config snapshot
    status_cb:         callable        # fn(str) → None
    name_matcher:      NameMatcher     # per-scan NameMatcher instance
    logo_cache:        list            # [Box | None] — last logo position
    last_name_shot:    list            # [PIL.Image | None]
    correction_event:  threading.Event
    correction_result: list            # [str | None]
    correction_cb:     callable        # fn(raw) → triggers dialog on main thread
    slot_results:      list            # list[SlotData]
    players_dict:      dict            # player → slots placed count
    ml_model_ready:    object = None   # None = unchecked, bool after first check
```

---

## Coordinate System (CRITICAL)

- **All baseline coordinates are 1920×1080.**
- Constants in `_HUD`, `_TEAM_BUILDINGS_BL`, button positions are ALL 1920×1080 baseline.
- Convert to live screen coords via `win.sp(bx, by)` (point) or `win.sr(bx, by, bw, bh)` (region).
- Named HUD regions: `win.hud("timer")` — looks up `_HUD["timer"]` and calls `win.sr()`.
- **NEVER hardcode live pixel coordinates.** Always derive from `Window` methods.
- `Window(x, y, w, h)` stores the game canvas top-left + dimensions.
  - `win.sx = w / 1920`, `win.sy = h / 1080` — scale factors.

```python
# Correct
pos = win.sp(480, 843)          # scales _NEXT_BTN_BL baseline to live coords
region = win.hud("timer")       # scales named HUD region

# Wrong — never do this
pos = (480, 843)                # raw baseline — won't work at non-1920 resolutions
```

---

## Config Structure (config.json)

```json
{
    "game_window_title":     "BlueStacks",
    "content_offset_x":     0,
    "content_offset_y":     0,
    "tesseract_path":       "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
    "players": {
        "JAYSEN":   "UTC+1",
        "KOIVU":    "UTC+2"
    },
    "ocr_corrections": {
        "鮭とば":          "SAKETOBA",
        "ぴよきち":        "PIYO",
        "PATRICKSPATZ":   "PATRICK"
    },
    "enemy_buildings": {
        "1": [1473, 507],
        "2": [1498, 778]
    },
    "name_match_cutoff":     0.80,
    "bot_slots_total":       75,
    "sleep_start":           21,
    "sleep_end":             8,
    "cjk_names":             false,
    "max_enemy_slots":       14,
    "discord_webhook_url":   "",
    "scan_interval_minutes": 0,
    "auto_train":            true,
    "alert_thresholds": {
        "instant_warning_minutes": 30,
        "building_strength_min":   5000000,
        "score_gap_warning":       10000
    },
    "alert_webhook_url":     ""
}
```

**Adding a player:** Edit `players` dict. Config reloads at scan start — no restart needed.

---

## OCR Pipeline

### Score / Timer OCR

```
screenshot(region) → converter(image) → upscale 2× NEAREST → MinFilter(3) → pad 10px → Tesseract
```

- `convert_to_bw` — near-white (R≥235,G≥242,B≥254) → black ink (white HUD text)
- `convert_dark_text` — dark pixels (luminance < 170) → black ink (scores, opponent name)
- `convert_white_text` — generic white text → black ink (max_points)
- `convert_badge_text` — white pixels (≥180 all channels) → black ink (bonus badges)

### Build Image Stats OCR

```
build_card.PNG → crop bottom 14% → dark-text binarize → gap detection → split HP|ATK → Tesseract
```

- Gap-based: finds widest gap between digit groups to separate HP and ATK sections
- Skips heart/sword icons via gap detection (no color matching needed)
- Results cached in `_stats_cache.json` by file modification time

### Player Name OCR (in priority order)

1. **Template match** — NameMatcher, fastest path (no Tesseract/ML needed)
2. **CNN classifier** — `name_model.predict_name()`, fast ~5ms when model is trained
3. **ASCII Tesseract PSM 7** — `_NAME_OCR` config, corrections applied
4. **ASCII Tesseract PSM 8** — single-word mode, corrections applied
5. **CJK Tesseract** — `_NAME_OCR_CJK`, Otsu binarize, `_resolve_cjk()` cleanup
6. **EasyOCR fallback** — handles complex scripts, slower
7. **Low-cutoff fuzzy match** — last-resort matching at reduced threshold
8. **Correction dialog** — user resolves, saves to config + template

On every successful name match (any method), a training sample is auto-saved via
`name_model.save_training_sample()` for future CNN training.

### Paw Icon Blanking (IMPORTANT)

When blanking the paw icon from the name region, use **160 (mid-gray)**, NOT 255.
Pure white (255) triggers `convert_to_bw`'s near-white threshold and becomes black ink,
producing spurious OCR characters (e.g. 'C') from the blanked area.

```python
arr[:, :card.paw_end] = 160   # correct
arr[:, :card.paw_end] = 255   # WRONG — creates black ink in processed image
```

---

## Key Pixel Colour Rules (team building map icons)

| Pixel colour | Meaning |
|---|---|
| `(255,101,100)` or `(94,106,131)` | Building empty — skip |
| `(144,179,235)` or `(0,0,0)` | Loading / placeholder — keep clicking |
| `(169,202,250)` | First slot indicator — wrapped back to start |
| `(255,255,255)` | Next button white — ready to click |
| green ≥ red (status px) | DEFENDING |
| red > green (status px) | ATTACKING |

---

## Template Images Required (images/)

| File | Used for |
|---|---|
| `lobby.PNG` | Detect lobby screen (DEFEND/ATTACK choice) |
| `team_logo.png` | Locate player card, derive name region |

DEFEND and ATTACK buttons use baseline coords (`_DEFEND_BTN_BL`, `_ATTACK_BTN_BL`)
in `templates.py` — no template images needed for buttons.

---

## Debugging

All debug images saved to `images/`:

| File | What it shows |
|---|---|
| `debug_window.PNG` | Full game window at scan start |
| `debug_name_raw.PNG` | Raw name region screenshot |
| `debug_name_conv.PNG` | Name after convert_to_bw + upscale |
| `debug_name_cjk.PNG` | Name after Otsu binarize (CJK path) |
| `debug_timer_raw.PNG` | Timer region screenshot |
| `debug_timer_conv.PNG` | Timer after processing |
| `debug_slot_hp_raw.PNG` | HP region screenshot (first slot only) |
| `debug_slot_atk_raw.PNG` | ATK region screenshot (first slot only) |
| `debug_opponent_raw.PNG` | Opponent name region screenshot |
| `debug_opponent_conv.PNG` | Opponent name after processing |
| `crash.log` | Full traceback on unhandled exception |

**When OCR returns wrong values:** Check `_raw` file first. If region is wrong, adjust
`_HUD` baseline coords. If region is right but text is unreadable, check `_conv` file
to see what Tesseract receives.

---

## Logging Infrastructure

All modules use Python's `logging` module via `get_logger(name)` from `bot/config.py`.
No `print()` statements remain in the codebase.

```python
from bot.config import get_logger
_log = get_logger("module_name")   # returns logging.getLogger("bot.module_name")
```

**Output:** Dual — rotating file (`bot.log`, 2MB max, 3 backups) + console (INFO+).
**Log levels:**

- `DEBUG` — verbose OCR details, cache hits, fingerprint diffs
- `INFO` — scan progress, slot captures, matches, training results
- `WARNING` — suspicious values, failed OCR, rejected garbage, slot count anomalies
- `ERROR` — on_slot crashes (with `exc_info=True`), impossible player counts
- `CRITICAL` — unhandled exceptions in main crash handler

**Log file:** `bot.log` in project root (`LOG_PATH` in config.py).

---

## OCR Validation & Sanity Checks

- **HP/ATK minimum:** Live OCR values < 1000 are rejected as garbage (set to 0).
  CNN sometimes returns tiny values like "2" — these are logged as warnings.
- **Timer garbage:** All-same-digit strings (e.g. '11111') from CNN are rejected.
- **Per-building slot cap:** Warning logged when a player has > 3 cars in one building
  (max possible is 3 — indicates wrap detection failure).
- **Total scan cap:** Warning when total slots > `BOT_SLOTS_TOTAL` (75).
- **Per-player cap:** Error when > 18 slots (impossible — 6 buildings × 3 max),
  warning when > 6 (suspicious).

---

## Performance Invariants

- `pyautogui.PAUSE = 0` and `MINIMUM_DURATION = 0` — set in `navigation.py`, never change.
- Cache all `win.hud()` regions at scan start — don't call in a loop.
- `_lobby_visible()` is throttled to at most once per 500ms in slot loops.
- Logo template search uses cached position (±20px region) for slots 2+ per building.
- Score OCR: 5 values captured in parallel via `ThreadPoolExecutor(max_workers=5)`.
- Build OCR: all saved screenshots OCR'd in parallel via `ThreadPoolExecutor(max_workers=6)` after slot loop.
- Build stats cached by file mtime in `_stats_cache.json` — only re-OCR'd when image changes.
- `_advance_slot` polls a 40×40 center crop, not the full build region.
- ML models use lazy imports (`import torch` inside functions) to avoid ~3s startup cost.
- **Pre-load OCR model before ThreadPoolExecutor** — concurrent `import torch` across 5 workers
  causes deadlock/stall (>15s timeout). `_load_model()` is called on the scan thread first.
- `_auto_train_all()` runs in a background daemon thread after each scan — never blocks scan completion.
- Training is skipped if sample count hasn't changed since last run (`_last_train_count`).
- Name model skips retrain when class set unchanged (`_classes_unchanged()` helper).
- ML model state (`_model`, `_classes`) protected by `_model_lock` for scan + train thread safety.
- Early stopping (patience=10) + ReduceLROnPlateau scheduler in both ML training loops.

---

## ML Training Data

```
training_data/
├── names/                  ← name-region screenshots, one dir per player
│   ├── JAYSEN/             ← {timestamp}.png files (cap: 50 per player)
│   ├── KOIVU/
│   └── ...
├── ocr/                    ← OCR training samples, one dir per field type
│   ├── timer/              ← {timestamp}_{value}_conv.png + _raw.png
│   ├── team_points/
│   ├── opp_points/
│   ├── team_bonus/
│   ├── opp_bonus/
│   └── max_points/
├── name_model.pth          ← trained CNN weights (name classifier)
├── name_classes.json       ← class index → player name mapping
├── ocr_char_model.pth      ← trained CNN weights (character classifier)
└── ocr_char_classes.json   ← class index → character mapping
```

**Auto-training triggers:** After each scan, `_auto_train_all()` checks if new samples
exist since last training. Name model needs ≥2 players with ≥3 samples each (≥10 total).
OCR model needs ≥20 total samples. Training runs in a background thread.

---

## Wrap Detection (navigation.py)

Slot cycling ends when the view wraps back to slot 1. Two detection criteria
(BOTH must match for a wrap):

1. **Same player name** as slot 1
2. **Similar car screenshot** — fingerprint diff < `WRAP_FP_THRESHOLD` (32×32 grayscale
   downscale, mean abs diff)

**HP/ATK is NOT used for wrap detection.** Live OCR is too noisy — the CNN returns
garbage values like "2" (rejected to 0) or wildly different readings like 2813 vs 6845
for the same slot. This caused buildings to loop 3–5x before wrapping (B3: 45 slots,
B4: 51 slots). The fingerprint comparison reliably distinguishes different cars
(diff 0–1 for same car vs 30+ for different cars).

Same name alone is NOT a wrap (a player can have up to 3 cars per building).
Other end conditions: ESC held, lobby visible, advance_fn failure after retry, stuck
detection (frame unchanged), max_slots reached.

Key constants (tuned empirically):

- `WRAP_FP_THRESHOLD = 5` — fingerprint diff below which = same car
- `SLOT_SETTLE_TIME = 0.20` — wait for new slot to fully render before comparison

---

## Build History (build_history/)

Stable 3-car record per player. After each scan's slot loop completes, `_update_build_history()`
copies up to 3 screenshots per player from `builds/` to `build_history/{PLAYER}/` using
`shutil.copy2` (not move). Files are renamed to `{PLAYER}_1.PNG` through `{PLAYER}_3.PNG`.
Any files numbered `_4` or higher are cleaned up.

This ensures `build_history/` always has the latest 3-car record regardless of how many
cars were deployed on the map during the current scan. The top players report
(`format_top_players()`) reads from `build_history/` so it always reflects all 3 cars.

```python
# Correct: copy from builds/ to build_history/ (preserves builds/ for next scan)
shutil.copy2(src_path, os.path.join(dst, f"{player}_{i}.PNG"))

# Wrong: move (destroys builds/ source)
shutil.move(src_path, dst_path)
```

---

## War Tracking (history.py)

- `get_or_create_war(opponent)` finds ongoing war or creates new one.
- **Auto-close**: when a new opponent is detected, all ongoing wars are automatically closed
  using the last score snapshot to determine win/loss/draw.
- Score snapshots saved per scan for trajectory analysis.
- Enemy player slots saved per war for opponent scouting.

---

## Anti-Patterns (Do Not)

```python
# ❌ Never hardcode live coords
pyautogui.click(480, 843)

# ❌ Never access tkinter from a background thread
self.label.config(text=msg)           # → AttributeError or crash

# ❌ Never sleep without a timeout + escape check
while True:
    time.sleep(0.1)
    if condition: break

# ❌ Never let _name_templates grow unbounded
_name_templates[player].append(new)   # always cap at 5

# ❌ Never swallow exceptions silently
try:
    ...
except Exception:
    pass                               # log it: _log.error("msg: %s", e)

# ❌ Never write scan state to module globals
_last_result = something               # use ScanContext instead

# ❌ Never call win.hud() inside a slot loop
for slot in slots:
    r = win.hud("slot_hp")             # compute once before loop

# ❌ Never import torch concurrently in ThreadPoolExecutor workers
# Pre-load on the scan thread BEFORE spawning workers:
from bot.ocr_model import is_model_ready, _load_model
if is_model_ready():
    _load_model()   # ← do this BEFORE ThreadPoolExecutor

# ❌ Never delete build_history/ — it's the stable 3-car record
shutil.rmtree(build_history_dir)      # builds/ is ephemeral; build_history/ is the record

# ❌ Never update _last_train_count outside _model_lock
_last_train_count = n                  # must be inside `with _model_lock:`
```

---

## Feature Status

| Feature | Module | Status |
|---|---|---|
| Modular bot/ package | all | ✅ v4.0 |
| ScanContext (no global state) | scan.py | ✅ v4.0 |
| Config hot-reload (mtime check) | config.py | ✅ v4.0 |
| SQLite scan history | history.py | ✅ v4.0 |
| Discord webhook post | discord_post.py | ✅ v4.0 |
| Auto-scan interval | WhatsUpBot.py | ✅ v4.0 |
| Universal dark-text OCR | vision.py, ocr.py | ✅ v4.1 |
| Build image stats (gap-based) | ocr.py, report.py | ✅ v4.1 |
| Parallel build OCR | scan.py | ✅ v4.1 |
| Mtime-based stats cache | report.py | ✅ v4.1 |
| Build image grid | report.py | ✅ v4.1 |
| HUD calibration mode | WhatsUpBot.py, templates.py | ✅ v4.1 |
| EasyOCR name recognition | ocr.py, easyocr_engine.py | ✅ v4.1 |
| System tray | WhatsUpBot.py | ✅ v4.1 |
| War tracking + score snapshots | history.py, scan.py | ✅ v5.0 |
| Opponent scouting DB | history.py, scan.py | ✅ v5.0 |
| Building strength analysis | analysis.py, report.py | ✅ v5.0 |
| Score momentum tracking | analysis.py, report.py | ✅ v5.0 |
| Placement recommendations | analysis.py, report.py | ✅ v5.0 |
| Configurable alerts | alerts.py, scan.py | ✅ v5.0 |
| Semantic wrap detection | navigation.py | ✅ v5.0 |
| EasyOCR safe imports | ocr.py | ✅ v5.0 |
| Parallel OCR timeout | scan.py | ✅ v5.0 |
| Correction dialog fix | WhatsUpBot.py, scan.py | ✅ v5.0 |
| CNN name classifier | name_model.py, scan.py | ✅ v5.0 |
| CNN OCR character model | ocr_model.py, scan.py | ✅ v5.0 |
| Auto-training after scans | scan.py | ✅ v5.0 |
| Training data auto-collection | name_model.py, scan.py | ✅ v5.0 |
| Build history preservation | scan.py, config.py | ✅ v5.0 |
| War auto-close on new opponent | history.py | ✅ v5.0 |
| ML early stopping + LR scheduler | name_model.py, ocr_model.py | ✅ v5.0 |
| Thread-safe model loading | name_model.py, ocr_model.py | ✅ v5.0 |
| OCR model pre-load (deadlock fix) | scan.py | ✅ v5.0 |
| Wrap detection confirmation | navigation.py | ✅ v5.0 |
| Smart retrain skip | scan.py | ✅ v5.0 |
| Macro recorder | TBD | 🔲 planned |

---

## Testing Strategy

```
tests/
├── test_ocr.py          # _resolve_cjk, _digits
├── test_report.py       # parse_timer, _fmt_stat, _fmt_hm, _tz_offset
├── test_vision.py       # convert_to_bw, convert_dark_text, convert_badge_text
├── test_history.py      # SQLite history, war tracking, score snapshots, opponent roster
├── test_analysis.py     # building analysis, momentum, recommendations, formatting
├── test_alerts.py       # alert evaluation, thresholds, config
├── test_config.py       # config loading, mtime check, derived constants
└── test_ocr_model.py    # character segmentation, label extraction, dataset building
```

Key test targets (no BlueStacks needed):

- `parse_timer("17h 23m")` → `(17, 23)`
- `parse_timer("0823")` → `(8, 23)` (no 'h')
- `_fmt_stat(1_234_567)` → `"1.2M"`
- `_resolve_cjk("し りぶとん")` → `"LIPTON"` (given correction in config)
- `segment_characters(img)` → list of character crops from conv image
- Report builder given mock `SlotData` list → expected string output
