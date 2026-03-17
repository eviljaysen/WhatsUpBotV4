# bot/ Package — Module Reference

> Per-module details for Claude Code. Read root CLAUDE.md first.

---

## bot/config.py

**Purpose:** Single source of truth for all configuration. Loaded once at startup, reloadable on demand.

**Exports:**
- `CFG: dict` — raw config dict
- `reload_config() -> dict` — reload config.json, update all derived constants, return new dict
- `GAME_WINDOW_TITLE`, `CONTENT_OFFSET_X/Y` — window detection params
- `player_timezones: dict` — name → "UTC±N"
- `_KNOWN_NAMES: list` — sorted list of all known player names
- `_NAME_UPPER_MAP: dict` — uppercase name → canonical name
- `_OCR_CORRECTIONS: dict` — raw OCR string → corrected player name
- `NAME_MATCH_CUTOFF`, `BOT_SLOTS_TOTAL`, `SLEEP_START/END` — scalar settings
- `_CJK_NAMES: bool` — whether CJK OCR path is active
- `_TMPL_THRESHOLD: float` — Jaccard IoU threshold for name template matching
- `_ENEMY_BLDGS_CFG: dict` — building number → [x, y] baseline coords
- `BASE_DIR`, `IMAGES_DIR`, `BUILDS_DIR`, `SCANS_DIR`, `NAME_TMPL_DIR` — paths
- `save_ocr_correction(raw: str, corrected: str)` — persist correction to config.json

**Key rule:** All other modules import constants FROM config. Config never imports from bot/.

---

## bot/window.py

**Purpose:** Detect the BlueStacks game window and expose scaled coordinate helpers.

**Exports:**
- `class Window` — geometry + scale helpers
  - `win.sp(bx, by) -> (x, y)` — scale 1920×1080 point to live screen
  - `win.sr(bx, by, bw, bh) -> (x, y, w, h)` — scale 1920×1080 region
  - `win.hud(key) -> (x, y, w, h)` — scale named HUD region
  - `win.region -> (x, y, w, h)` — full window region
- `detect_window() -> Window` — find BlueStacks, save debug_window.PNG

**How window detection works:**
1. `EnumWindows` finds all visible top-level windows matching `GAME_WINDOW_TITLE`
2. `EnumChildWindows` finds the largest child surface (render canvas)
3. `ClientToScreen` converts client coords to screen coords
4. Falls back to full screen if not found
5. Applies `CONTENT_OFFSET_X/Y` from config

---

## bot/templates.py

**Purpose:** Lazy-loading, auto-rescaling template image cache. Also owns the HUD coord map.

**Exports:**
- `_HUD_DEFAULTS: dict` — default HUD regions as 1920×1080 `(bx, by, bw, bh)` tuples
- `_HUD: dict` — active HUD regions (defaults + any user overrides from config)
- `reload_hud_overrides()` — re-apply HUD overrides from CFG after config reload
- `_TEAM_BUILDINGS_BL: dict` — building number → (bx, by) baseline
- Navigation baselines: `_ESCAPE_BL`, `_DEFEND_BTN_BL`, `_ATTACK_BTN_BL`, `_NEXT_BTN_BL`
- `class Templates` — image cache with auto-rescaling
  - `tpl.refresh(win)` — reload if scale changed
  - `tpl.get(filename) -> PIL.Image | None`
  - `tpl.find(filename, region, confidence) -> Box | None`
  - `tpl.find_center(filename, region, confidence) -> (x, y) | None`
- `get_tpl() -> Templates` — access the singleton

**HUD regions to calibrate (verify with debug images after first scan):**
- `slot_hp` — HP value row below car card
- `slot_armor` — armor/power value row below car card
- `slot_status_px` — 1px sample for DEF/ATK detection

---

## bot/vision.py

**Purpose:** Pure image processing — no OCR, no screen capture. All functions take PIL Images,
return PIL Images or numpy arrays.

**Exports:**
- `convert_to_bw(image, white_threshold=(235,242,254)) -> PIL.Image`
  - Near-white pixels → black ink; everything else → white background
  - Used for: white HUD text (timer, names)
- `convert_white_text(image) -> PIL.Image`
  - Generic white (≥200 all channels) → black ink
  - Used for: max_points region
- `convert_dark_text(image, threshold=170) -> PIL.Image`
  - Dark pixels (luminance < threshold) → black ink; light → white
  - Universal converter for any dark-on-light text regardless of color
  - Used for: team_points, opp_points, HP/ATK, opponent name
- `convert_badge_text(image) -> PIL.Image`
  - White pixels (all channels ≥ 180) → black ink
  - Used for: bonus badge +NNN text on colored backgrounds

**Critical: `convert_to_bw` threshold**
The threshold (235,242,254) is tuned for the game's HUD font — near-white but not pure white.
Don't change it without testing all affected OCR regions. The paw-blanking color (160) is
specifically chosen to NOT trigger this threshold.

---

## bot/ocr.py

**Purpose:** All OCR operations — Tesseract config, score reading, name recognition pipeline,
CJK support, name template matching.

**Exports:**
- OCR config strings: `_BONUS_OCR`, `_NAME_OCR`, `_NAME_OCR_PSM8`, `_NAME_OCR_CJK`,
  `_NUMBERS_OCR`, `_TIMER_OCR`
- `_ocr(image, config) -> str` — raw Tesseract call
- `_digits(text) -> int` — extract integer from OCR string
- `_prepare_for_ocr(img, converter, upscale, pad, minfilter) -> PIL.Image`
- `locate_number(region, converter, ocr_config, debug_name, upscale, minfilter) -> int`
- `_prep_name_image(shot) -> PIL.Image` — 3× upscale + B&W + MinFilter + pad
- `_read_name(proc, cfg) -> str` — OCR + ASCII normalise + corrections
- `_name_match(text) -> list` — fuzzy match against known names
- `_resolve_cjk(raw_cjk) -> str` — strip garbage, exact + fuzzy correction lookup
- `ocr_player_name(card, ctx) -> str` — full multi-pass name pipeline
- `ocr_build_stats(image_path) -> (hp, atk)` — gap-based HP/ATK extraction from build card images
- `_ocr_enemy_name(win) -> str` — enemy car panel name OCR
- `class NameMatcher` — manages name templates (load/save/match)
  - Per-scan instance in ScanContext (not global)
  - `match(shot) -> str` — IoU template match
  - `save_template(player, shot)` — save to name_templates/, cap at 5

**NameMatcher replaces v3's global `_name_templates` dict.**
Each `ScanContext` gets a fresh `NameMatcher` initialized from disk at scan start.

---

## bot/navigation.py

**Purpose:** All mouse/keyboard interaction with the game. Knows the game's navigation flow.
No OCR, no image processing — pure click/wait/detect.

**Exports:**
- `_esc() -> bool` — check if ESC is held
- `_lobby_visible(win, ctx, confidence) -> bool` — throttled lobby template check
- `_find_next_btn_pos(win) -> (x,y) | None` — locate white Next button cluster
- `_move_garage(win, ctx)` — click back until lobby template appears
- `_move_lobby(win, ctx)` — click back until lobby disappears
- `_move_enter(win, ctx, btn_bl)` — click DEFEND or ATTACK button until lobby disappears
- `_enter_building(win, ctx, pos) -> bool` — click building icon, return False if empty
- `_advance_slot(win, build_region) -> bool` — click Next, wait for card change
- `_cycle_slots(win, ctx, on_slot, advance_fn, build_region, check_lobby, max_slots)` — unified slot cycling

**Navigation flow (team and enemy use same flow):**
```
map screen
  → click building icon (_enter_building)
    → lobby screen (DEFEND / ATTACK choice)
      → _move_enter(btn_bl=_DEFEND_BTN_BL)  — team
      → _move_enter(btn_bl=_ATTACK_BTN_BL)  — enemy
        → slot view
          → _cycle_slots: on_slot → _advance_slot → semantic + frame-diff wrap detection
        → _move_garage → _move_lobby → back to map
```

---

## bot/scan.py

**Purpose:** Orchestrates full scans. Creates ScanContext, calls navigation + OCR + report.
This is the only module that knows the full scan flow.

**Exports:**
- `@dataclasses.dataclass SlotData` — data from one player slot
- `@dataclasses.dataclass ScanContext` — all mutable scan state
- `capture_slot_stats(win, ctx, build_path) -> (hp, atk, defending)` — OCR HP/ATK/status
- `run_team_scan(status_cb, correction_cb, avail_only) -> (report_str, meta_dict)`
- `run_enemy_scan(building_num, status_cb) -> str`

**run_team_scan flow:**
```
detect_window → refresh_templates → create ScanContext
  for bnum in 1..6:
    enter_building → cycle_slots → [on_slot: OCR name + screenshot + DEF/ATK status]
    exit_building
  parallel build OCR (ThreadPoolExecutor — all saved screenshots at once)
  capture scores (parallel ThreadPoolExecutor)
  capture timer
  build_report(ctx) → return (report, meta) [includes analysis sections]
  save to SQLite + war tracking + score snapshot
  evaluate alerts → fire to Discord if thresholds met
```

**on_slot responsibilities:**
1. OCR player name
2. Validate against known players, check slot count (max 3)
3. Save build screenshot
4. Capture DEF/ATK status pixel (HP/ATK deferred to parallel batch)
5. Append SlotData to ctx.slot_results

---

## bot/report.py

**Purpose:** Report building, formatting, and player stats extraction.
Builds formatted reports from scan data and extracts player stats from
saved build images via OCR.

**Exports:**
- `_fmt_hm(minutes: int) -> str` — `"17H 23m"`
- `_fmt_stat(n: int) -> str` — `"2.2M"`, `"639K"`, `"—"`
- `parse_timer(text: str) -> (hours, minutes)`
- `_tz_offset(tz_str: str) -> int` — `"UTC+2"` → `2`
- `build_report(ctx, scores, h, m, avail_only) -> (report_str, meta_dict)`
  - Builds the full Discord-formatted report string
  - Player stats table (sorted by HP desc)
  - Bot availability (awake/asleep split)
  - Win/loss instant projection
- `get_player_stats_from_builds() -> list` — OCR HP/ATK from saved build images (mtime-cached)
- `format_player_stats_table(stats) -> str` — format stats list into readable table
- `build_image_grid(screenshot_paths, cols) -> PIL.Image` — grid collage of build screenshots

---

## bot/history.py (v5.0)

**Purpose:** Persist scan results, war tracking, score trajectories, and opponent
scouting data to SQLite.

**DB Tables:** `scans`, `slots`, `player_stats`, `wars`, `score_snapshots`, `opponent_players`

**Exports (v4.0):**
- `init_db()` — create tables if not exist
- `save_scan(meta, slot_results) -> int` — returns scan_id
- `save_player_stats(scan_id, timestamp, player, slots)` — aggregated player strength
- `get_latest_player_stats() -> list` — most recent stats per player
- `get_player_stats_history(player, limit) -> list`
- `get_last_scan(opponent) -> (meta, slots) | None`
- `get_player_hp_trend(player, limit=5) -> list[int]`
- `get_opponent_history(opponent, limit) -> list`

**Exports (v5.0 — war tracking):**
- `get_or_create_war(opponent) -> int` — find ongoing war or create new
- `save_score_snapshot(scan_id, war_id, meta)` — point-in-time score
- `get_score_trajectory(war_id) -> list` — all snapshots for a war (ASC)
- `close_war(war_id, result, our_final, opp_final)` — mark war complete
- `save_enemy_slot(war_id, opponent, player_name, building, ...)` — scouting data
- `get_opponent_roster(opponent) -> list` — latest per player (by strength DESC)
- `get_war_history(limit=20) -> list` — recent wars with results
- `get_war_by_id(war_id) -> dict | None`

---

## bot/analysis.py (v5.0)

**Purpose:** Pure strategic analysis — no screen capture, no OCR, no pyautogui.

**Exports:**
- `analyze_buildings(slot_results) -> dict` — per-building totals, weakest/strongest/empty
- `recommend_placements(slot_results, player_slots_placed, player_stats) -> list`
  - Returns `[(player, building, reason), ...]` — strongest unplaced → weakest buildings
- `get_momentum(snapshots) -> dict` — velocity, acceleration, trend per team
- `format_building_summary(analysis) -> str` — compact table
- `format_recommendations(recommendations) -> str` — placement recommendations
- `format_momentum(momentum) -> str` — one-line "Team +N/min [trend] | Opp +N/min [trend]"

---

## bot/alerts.py (v5.0)

**Purpose:** Evaluate scan results against configurable thresholds, fire alerts to Discord.

**Alert types:** `INSTANT_WARNING`, `BUILDING_WEAK`, `SCORE_GAP_CLOSING`, `CARS_LOST`

**Exports:**
- `class Alert` — dataclass: alert_type, severity, message
- `class AlertEvaluator`
  - `__init__(cfg)` — reads `alert_thresholds` from config
  - `evaluate(meta, analysis, trajectory, prev_slot_count) -> list[Alert]`
  - `fire(alerts)` — post to Discord via `post_alert()`

**Config keys:**
```json
{
    "alert_thresholds": {
        "instant_warning_minutes": 30,
        "building_strength_min": 5000000,
        "score_gap_warning": 10000
    },
    "alert_webhook_url": ""
}
```

---

## bot/discord_post.py (v4.0)

**Purpose:** Post scan reports to Discord via webhook. No dependencies beyond stdlib.

**Exports:**
- `post_report(webhook_url, report_text, image_path=None) -> bool`
  - POSTs text as a Discord message embed
  - Optionally attaches a build image grid
  - Returns True on success, False on failure
- `post_alert(webhook_url, message) -> bool` — post a short alert message

**Implementation note:** Use `urllib.request.urlopen` — no `requests` dependency needed.
Payload: `{"content": report_text}` for text, multipart for image attachment.

---

## ScanContext Fields Reference

```python
@dataclasses.dataclass
class ScanContext:
    win:               Window           # live game window
    cfg:               dict             # config snapshot for this scan
    status_cb:         callable         # fn(str) → progress update to UI
    name_matcher:      NameMatcher      # loaded from disk at scan start
    logo_cache:        list             # [Box|None] — reset per building
    last_name_shot:    list             # [PIL.Image|None] — for correction dialog
    correction_event:  threading.Event  # signals correction dialog completed
    correction_result: list             # [str|None] — result from dialog
    correction_cb:     callable         # fn(raw) → triggers dialog on main thread
    slot_results:      list             # [SlotData] — accumulates during scan
    players_dict:      dict             # player → slot count placed this scan
```

---

## SlotData Fields Reference

```python
@dataclasses.dataclass
class SlotData:
    player:     str    # canonical player name
    building:   int    # building number 1–6
    hp:         int    # HP value (0 = OCR failed / not captured)
    atk:        int    # attack value (0 = OCR failed)
    defending:  bool   # True = DEFENDING, False = ATTACKING
    screenshot: str    # absolute path to saved build PNG
```

---

## Common Patterns

### Pattern: Throttled visibility check
```python
_lobby_t = 0.0
while ...:
    now = time.time()
    if now - _lobby_t > 0.5:
        if _lobby_visible(win, ctx): break
        _lobby_t = now
```

### Pattern: Pixel-change wait with timeout
```python
pre = np.asarray(pyautogui.screenshot(region=region))
deadline = time.time() + timeout_s
while time.time() < deadline and not _esc():
    time.sleep(0.05)
    cur = np.asarray(pyautogui.screenshot(region=region))
    if not np.array_equal(pre, cur):
        break
```

### Pattern: Thread-safe UI update
```python
# In a background thread:
ctx.status_cb(f"Building {n}/6…")      # calls root.after(0, ...) internally
```

### Pattern: Safe file path from player name
```python
safe = re.sub(r'[\\/:*?"<>|]', '', player)
path = os.path.join(BUILDS_DIR, safe, f"{safe}_{count}.PNG")
```
