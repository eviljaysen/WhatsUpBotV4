"""bot/scan.py — Scan orchestration.

Creates ScanContext, coordinates navigation + OCR + report building.
This module owns the full scan flow but delegates all details to other modules.

Exports:
    CardInfo       — player card geometry
    SlotData       — data captured from one player slot
    ScanContext    — all mutable state for one scan run
    find_player_card(win, ctx) -> CardInfo
    capture_slot_stats(win, ctx) -> (hp, atk, defending)
    run_team_scan(cfg, status_cb, correction_cb) -> (report_str, meta_dict)
    run_enemy_scan(building_num, cfg, status_cb) -> str
"""

import dataclasses
import os
import re
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyautogui

from bot.config import (
    IMAGES_DIR, BUILDS_DIR, SCANS_DIR,
    player_timezones, _KNOWN_NAMES,
    BOT_SLOTS_TOTAL, CFG,
    reload_config
)
from bot.window import detect_window
from bot.templates import get_tpl, _TEAM_BUILDINGS_BL, _HUD
from bot.vision import convert_dark_text, convert_white_text, convert_badge_text
from bot.ocr import (
    NameMatcher, ocr_player_name, _ocr_enemy_name,
    locate_number, _ocr,
    _BONUS_OCR, _NUMBERS_OCR
)
from bot.navigation import (
    _esc, _lobby_visible, _enter_building,
    _advance_slot, _cycle_slots,
    _move_lobby, _move_garage,
)
from bot.templates import _ATTACK_BTN_BL
from bot.report import build_report


# ── Data structures ────────────────────────────────────────────────────────────
class CardInfo:
    """Player card geometry for one slot, as detected on screen."""

    def __init__(self, logo_box, name_region, build_region, paw_end=0):
        self.logo_box     = logo_box       # pyautogui Box or None (fallback)
        self.name_region  = name_region    # (x, y, w, h) for name OCR
        self.build_region = build_region   # (x, y, w, h) for build screenshot
        self.paw_end      = paw_end        # pixels from shot-left to blank


@dataclasses.dataclass
class SlotData:
    """All data captured from a single player slot during a team scan."""
    player:     str
    building:   int
    hp:         int  = 0      # ♥ value (0 = OCR failed)
    atk:        int  = 0      # ⚔ attack value (0 = OCR failed)
    defending:  bool = True   # True = DEFENDING, False = ATTACKING
    screenshot: str  = ''     # absolute path to saved build PNG


@dataclasses.dataclass
class ScanContext:
    """All mutable state for one scan run. Passed through the call chain.

    Replaces v3's module-level mutables (_logo_cache, _last_name_shot, etc.).
    Create one per scan via ScanContext.create().
    """
    win:               object           # Window
    cfg:               dict
    status_cb:         object           # callable(str)
    name_matcher:      NameMatcher
    logo_cache:        list             # [Box | None]
    last_name_shot:    list             # [PIL.Image | None]
    correction_event:  threading.Event
    correction_result: list             # [str | None]
    correction_cb:     object           # callable(str) | None
    slot_results:      list             # [SlotData]
    players_dict:      dict             # player → slots placed
    ml_model_ready:    object = None    # None = unchecked, bool after first check

    @classmethod
    def create(cls, win, cfg, status_cb, correction_cb=None):
        ctx = cls(
            win=win,
            cfg=cfg,
            status_cb=status_cb,
            name_matcher=NameMatcher(),
            logo_cache=[None],
            last_name_shot=[None],
            correction_event=threading.Event(),
            correction_result=[None],
            correction_cb=correction_cb,
            slot_results=[],
            players_dict={p: 0 for p in player_timezones},
        )
        # Wrap correction_cb so it carries a reference to ctx.
        # The dialog on the main thread reads _ctx to signal the result.
        if correction_cb is not None:
            original = correction_cb
            def _wrapped(raw, _ctx=ctx, _orig=original):
                _wrapped._ctx = _ctx      # accessible by the dialog
                _orig(raw)
            _wrapped._ctx = None
            ctx.correction_cb = _wrapped
        return ctx

    def status(self, msg: str):
        print(msg)
        if self.status_cb:
            self.status_cb(msg)


# ── Player card detection ──────────────────────────────────────────────────────
def find_player_card(win, ctx) -> CardInfo:
    """Locate the player card in the left panel and return a CardInfo.

    Uses logo_cache from ctx for fast-path slot 2+ searches.
    Falls back to fixed-offset approach if template missing or not found.
    """
    tpl    = get_tpl()
    build_r = win.hud("build")

    img = tpl.get("team_logo.png")
    if img is not None:
        box = None

        cached = ctx.logo_cache[0]
        if cached is not None:
            from bot.navigation import LOGO_CACHE_MARGIN_PX
            margin = round(LOGO_CACHE_MARGIN_PX * win.sx)
            fast_r = (cached.left  - margin, cached.top    - margin,
                      cached.width + margin * 2, cached.height + margin * 2)
            box = tpl.find("team_logo.png", region=fast_r, confidence=0.75)

        if box is None:
            panel = (win.x, win.y, round(win.w * 0.35), win.h)
            box   = tpl.find("team_logo.png", region=panel, confidence=0.75)

        if box is not None:
            ctx.logo_cache[0] = box
            panel_right = win.x + round(win.w * 0.35)
            name_x  = box.left
            paw_end = box.width + max(4, round(10 * win.sx))
            name_w  = max(10, min(panel_right - name_x - 4, round(220 * win.sx)))
            name_h  = max(6, round(box.height * 2 / 3))
            name_r  = (name_x, box.top, name_w, name_h)
            return CardInfo(box, name_r, build_r, paw_end=paw_end)

    # Fallback: fixed paw offset
    paw_off  = round(66 * win.sx)
    logo_reg = win.sr(130, 290, 416, 160)
    name_r   = (logo_reg[0] + paw_off,
                logo_reg[1],
                logo_reg[2] - paw_off,
                round(logo_reg[3] * 2 / 3))
    return CardInfo(None, name_r, build_r)


# ── Slot stats capture ─────────────────────────────────────────────────────────
def capture_slot_stats(win, ctx, build_path: str = '') -> tuple:
    """Capture HP, attack, and defending/attacking status from the current slot.

    If build_path is provided, OCRs HP/ATK from the saved build screenshot
    (more reliable — gap-based dark-text extraction from the stat line).
    Falls back to HUD region OCR if build_path is empty or OCR fails.

    Returns (hp: int, atk: int, defending: bool).
    HP and atk are 0 when OCR fails. defending defaults to True on failure.
    """
    hp = atk = 0
    defending = True

    # Primary: OCR from saved build screenshot (proven accurate)
    if build_path:
        try:
            from bot.ocr import ocr_build_stats
            hp, atk = ocr_build_stats(build_path)
        except Exception as e:
            print(f"[stats] build OCR failed: {e}")

    # Fallback: HUD region OCR if build OCR returned nothing
    if hp == 0 and atk == 0:
        try:
            hp  = locate_number(win.hud("slot_hp"),  converter=convert_dark_text,
                                debug_name="slot_hp")
            atk = locate_number(win.hud("slot_atk"), converter=convert_dark_text,
                                debug_name="slot_atk")
        except Exception as e:
            print(f"[stats] HUD OCR failed: {e}")

    # Status pixel check (always from live screen)
    try:
        px = pyautogui.screenshot(region=win.hud("slot_status_px")).getpixel((0, 0))
        defending = px[1] >= px[0]   # green channel ≥ red → DEFENDING
    except Exception:
        pass
    return hp, atk, defending


# ── Team Scan ──────────────────────────────────────────────────────────────────
def run_team_scan(status_cb=None, correction_cb=None, avail_only=False) -> tuple:
    """Full team scan: player builds + scores + instant times.

    Args:
        status_cb: callable(str) for progress updates to UI
        correction_cb: callable(str) that triggers the correction dialog on main thread

    Returns:
        (report_string, metadata_dict)
    """
    cfg = reload_config()
    win = detect_window()
    get_tpl().refresh(win)

    ctx = ScanContext.create(win, cfg, status_cb, correction_cb)
    start_time = time.time()

    # ── Slot scan ─────────────────────────────────────────────────────────────
    for bnum in range(1, 7):
        if _esc():
            break
        ctx.status(f"Building {bnum}/6…")

        pos   = win.sp(*_TEAM_BUILDINGS_BL[bnum])
        empty = not _enter_building(win, ctx, pos)
        if empty:
            continue

        ctx.logo_cache[0] = None   # reset per-building card-position cache

        def _on_slot(slot, _bnum=bnum):
            """Process one slot. Returns player name for wrap detection."""
            card   = find_player_card(win, ctx)
            player = ocr_player_name(card, ctx)
            if not player or player not in player_timezones:
                if player:
                    print(f"[B{_bnum}] Unknown player '{player}' — skipping")
                return player or None
            count = ctx.players_dict.get(player, 0)
            if count >= 3:
                return player

            count += 1
            ctx.players_dict[player] = count
            ctx.status(f"  B{_bnum} — {player} ({count}/3)")

            # Auto-collect training sample for ML name model
            if ctx.last_name_shot[0] is not None:
                from bot.name_model import save_training_sample
                save_training_sample(player, ctx.last_name_shot[0])

            safe = re.sub(r'[\\/:*?"<>|]', '',
                          "GUMSO" if player == "|-_-J" else player)
            pdir = os.path.join(BUILDS_DIR, safe)
            if count == 1:
                shutil.rmtree(pdir, ignore_errors=True)
            os.makedirs(pdir, exist_ok=True)
            scr_path = os.path.join(pdir, f"{safe}_{count}.PNG")
            pyautogui.screenshot(region=card.build_region).save(scr_path)

            # Only capture defending status from live screen pixel here.
            # HP/ATK OCR deferred to parallel batch after all buildings.
            defending = True
            try:
                px = pyautogui.screenshot(
                    region=win.hud("slot_status_px")).getpixel((0, 0))
                defending = px[1] >= px[0]
            except Exception:
                pass
            ctx.slot_results.append(SlotData(
                player=player, building=_bnum,
                hp=0, atk=0, defending=defending,
                screenshot=scr_path,
            ))
            return player

        slots_before = len(ctx.slot_results)
        visited = _cycle_slots(win, ctx, _on_slot, _advance_slot,
                               win.hud("build"), check_lobby=True)
        slots_after = len(ctx.slot_results)
        captured = slots_after - slots_before
        print(f"[B{bnum}] visited={visited} captured={captured}")
        if captured < visited:
            print(f"[B{bnum}] WARNING: visited {visited} slots but only "
                  f"captured {captured} — some names may have failed OCR")
        _move_lobby(win, ctx)

    # ── Parallel build OCR (HP/ATK from saved screenshots) ───────────────────
    slots_to_ocr = [s for s in ctx.slot_results if s.screenshot]
    if slots_to_ocr:
        ctx.status(f"OCR'ing {len(slots_to_ocr)} build screenshots…")
        from bot.ocr import ocr_build_stats

        def _ocr_slot(slot):
            try:
                return ocr_build_stats(slot.screenshot)
            except Exception as e:
                print(f"[stats] build OCR failed for {slot.player}: {e}")
                return (0, 0)

        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(_ocr_slot, s): s for s in slots_to_ocr}
            for fut in futures:
                slot = futures[fut]
                try:
                    hp, atk = fut.result(timeout=30)
                except Exception as e:
                    print(f"[stats] build OCR timeout/error for {slot.player}: {e}")
                    hp, atk = 0, 0
                slot.hp = hp
                slot.atk = atk

    # ── Score / timer capture ─────────────────────────────────────────────────
    ctx.status("Capturing scores…")

    from bot.ocr import _TIMER_OCR
    from bot.vision import convert_to_bw
    from bot.report import parse_timer
    from PIL import ImageFilter

    time.sleep(0.5)   # let map screen settle after building exits

    timer_raw = ''
    shot = timer_up = None
    deadline = time.time() + 6.0
    while not _esc() and time.time() < deadline:
        shot     = pyautogui.screenshot(region=win.hud("timer"))
        timer_bw = convert_to_bw(shot)
        timer_bw = timer_bw.filter(ImageFilter.MinFilter(3))
        from PIL import Image as Img
        timer_up  = timer_bw.resize(
            (timer_bw.width * 3, timer_bw.height * 3), Img.NEAREST)
        # Gap-based edge trim: crop to actual text content bounds
        arr = np.asarray(timer_up).copy()
        col_dark = np.sum(arr < 128, axis=0)
        nz = np.where(col_dark > 0)[0]
        if len(nz) > 0:
            pad_cols = max(3, arr.shape[1] // 40)
            left = max(0, nz[0] - pad_cols)
            right = min(arr.shape[1], nz[-1] + 1 + pad_cols)
            arr = arr[:, left:right]
        padded = Img.fromarray(np.pad(arr, 20, constant_values=255))
        raw = _ocr(padded, config=_TIMER_OCR).strip()
        print(f"[timer] attempt raw={raw!r}")
        if any(c.isdigit() for c in raw):
            timer_raw = raw.strip("m\n ")
            break
        time.sleep(0.2)

    if shot:
        shot.save(os.path.join(IMAGES_DIR, "debug_timer_raw.PNG"))
    if timer_up:
        timer_up.save(os.path.join(IMAGES_DIR, "debug_timer_conv.PNG"))

    h, m = parse_timer(timer_raw)
    print(f"[timer] raw='{timer_raw}'  →  h={h} m={m}")
    if h == 0 and m == 0:
        print(f"[WARNING] Timer OCR failed — raw: '{timer_raw}'")

    # Auto-save timer training sample
    if (h > 0 or m > 0) and shot:
        from bot.ocr import _save_ocr_training_sample
        _save_ocr_training_sample("timer", shot, timer_up or shot, h * 100 + m)

    opp_shot = pyautogui.screenshot(region=win.hud("opponent"))
    opp_shot.save(os.path.join(IMAGES_DIR, "debug_opponent_raw.PNG"))
    from bot.ocr import _prepare_for_ocr
    opp_proc = _prepare_for_ocr(opp_shot, convert_dark_text, upscale=2, minfilter=3)
    opp_proc.save(os.path.join(IMAGES_DIR, "debug_opponent_conv.PNG"))
    opponent = _ocr(opp_proc, config='--oem 1 --psm 7').strip()
    if not opponent:
        opponent = _ocr(opp_proc, config='--oem 1 --psm 8').strip()
    if not opponent:
        try:
            from bot.easyocr_engine import read_text
            easy_opp, easy_conf = read_text(opp_shot)
            if easy_opp and easy_conf > 0.3:
                opponent = easy_opp
                print(f"[opponent] EasyOCR fallback: '{opponent}' (conf={easy_conf:.2f})")
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=5) as ex:
        f_max = ex.submit(locate_number,
                          win.hud("max_points"), convert_white_text,
                          "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789",
                          "max_points", 2)
        f_op  = ex.submit(locate_number, win.hud("opp_points"),
                          convert_dark_text, _NUMBERS_OCR, "opp_points", 2)
        f_ob  = ex.submit(locate_number, win.hud("opp_bonus"),
                          convert_badge_text,
                          _BONUS_OCR, "opp_bonus", 3, 0)
        f_tp  = ex.submit(locate_number, win.hud("team_points"),
                          convert_dark_text, _NUMBERS_OCR, "team_points", 2)
        f_tb  = ex.submit(locate_number, win.hud("team_bonus"),
                          convert_badge_text,
                          _BONUS_OCR, "team_bonus", 3, 0)
        max_points  = f_max.result()
        opp_points  = f_op.result()
        opp_bonus   = f_ob.result()
        team_points = f_tp.result()
        team_bonus  = f_tb.result()

    max_points = (max_points + 500) // 1000 * 1000

    print(f"[raw] team_bonus={team_bonus} opp_bonus={opp_bonus} "
          f"team_points={team_points} opp_points={opp_points} max_points={max_points}")

    # Sanity warnings — log suspicious values but don't silently alter them
    if team_bonus > 400:
        print(f"[WARNING] team_bonus={team_bonus} looks too high — check debug_team_bonus_raw.PNG")
    if opp_bonus > 400:
        print(f"[WARNING] opp_bonus={opp_bonus} looks too high — check debug_opp_bonus_raw.PNG")
    if team_points > 145000:
        print(f"[WARNING] team_points={team_points} looks too high — check debug_team_points_raw.PNG")
    if opp_points > 145000:
        print(f"[WARNING] opp_points={opp_points} looks too high — check debug_opp_points_raw.PNG")
    if max_points == 0:
        print("[WARNING] max_points OCR failed — check debug_max_points_raw.PNG")

    elapsed = int(time.time() - start_time)
    ctx.status(f"Done in {elapsed // 60}m{elapsed % 60}s")

    scores = dict(
        team_points=team_points, opp_points=opp_points,
        team_bonus=team_bonus,   opp_bonus=opp_bonus,
        max_points=max_points,   opponent=opponent,
    )
    report_str, meta = build_report(ctx, scores, h, m, avail_only=avail_only)
    print(report_str)

    # Persist scan + player stats to SQLite
    meta["timer_h"] = h
    meta["timer_m"] = m
    _save_scan_to_db(ctx, meta)

    # Evaluate and fire alerts (v5.0)
    _evaluate_alerts(ctx, meta)

    # Auto-train all models after scan
    if CFG.get("auto_train", True):
        _auto_train_all(ctx)

    return report_str, meta


def _auto_train_all(ctx):
    """Train all ML models using data collected during scans."""

    # 1. Name CNN classifier
    try:
        from bot.name_model import get_training_stats, train_model
        stats = get_training_stats()
        total = sum(stats.values())
        players_with_enough = sum(1 for c in stats.values() if c >= 3)
        if players_with_enough >= 2 and total >= 10:
            ctx.status("Training name model…")
            result = train_model(min_samples=3)
            if "error" not in result:
                ctx.status(f"Name model: {result['accuracy']:.0%} acc, "
                           f"{result['num_classes']} players")
            else:
                print(f"[train] Name model skipped: {result['error']}")
        else:
            print(f"[train] Name model: need more data "
                  f"({total} samples, {players_with_enough} players with >=5)")
    except Exception as e:
        print(f"[train] Name model error: {e}")

    # 2. OCR training data summary
    try:
        ocr_train_dir = os.path.join(
            os.path.dirname(IMAGES_DIR), "training_data", "ocr")
        if os.path.isdir(ocr_train_dir):
            fields = {}
            for field in os.listdir(ocr_train_dir):
                fdir = os.path.join(ocr_train_dir, field)
                if os.path.isdir(fdir):
                    count = len([f for f in os.listdir(fdir) if f.endswith("_raw.png")])
                    if count > 0:
                        fields[field] = count
            if fields:
                summary = ", ".join(f"{k}={v}" for k, v in sorted(fields.items()))
                print(f"[train] OCR samples collected: {summary}")
    except Exception as e:
        print(f"[train] OCR stats error: {e}")


def _save_scan_to_db(ctx, meta: dict):
    """Persist scan results, player stats, and war tracking to SQLite."""
    try:
        from collections import defaultdict
        from bot.history import (
            init_db, save_scan, save_player_stats,
            get_or_create_war, save_score_snapshot,
        )

        init_db()
        scan_id = save_scan(meta, ctx.slot_results)

        # War tracking: link scan to an ongoing war
        opponent = meta.get("opponent", "")
        if opponent:
            war_id = get_or_create_war(opponent)
            save_score_snapshot(scan_id, war_id, meta)

        # Group slots by player, save stats for players with all 3 cars
        player_slots = defaultdict(list)
        for s in ctx.slot_results:
            player_slots[s.player].append(s)

        saved = 0
        ts = int(time.time())
        for player, slots in player_slots.items():
            if len(slots) == 3 and all(s.hp > 0 for s in slots):
                save_player_stats(scan_id, ts, player, slots)
                saved += 1

        if saved:
            ctx.status(f"Saved stats for {saved} players to DB")
            print(f"[history] Player stats saved: {saved} players (3 cars each)")
        else:
            print(f"[history] No players with 3 cars + valid HP — stats not saved")
    except Exception as e:
        print(f"[history] DB save error: {e}")


def _evaluate_alerts(ctx, meta: dict):
    """Evaluate alert conditions and fire to Discord if thresholds met."""
    try:
        from bot.alerts import AlertEvaluator
        from bot.analysis import analyze_buildings

        evaluator = AlertEvaluator(ctx.cfg)
        analysis = analyze_buildings(ctx.slot_results)

        # Get trajectory for momentum-based alerts
        trajectory = []
        try:
            from bot.history import get_or_create_war, get_score_trajectory
            opponent = meta.get("opponent", "")
            if opponent:
                war_id = get_or_create_war(opponent)
                trajectory = get_score_trajectory(war_id)
        except Exception:
            pass

        # Get previous slot count for CARS_LOST detection
        prev_count = 0
        try:
            from bot.history import get_last_scan
            opponent = meta.get("opponent", "")
            if opponent:
                prev_meta, prev_slots = get_last_scan(opponent)
                if prev_slots:
                    prev_count = len(prev_slots)
        except Exception:
            pass

        alerts = evaluator.evaluate(meta, analysis, trajectory, prev_count)
        if alerts:
            ctx.status(f"⚠️ {len(alerts)} alert(s) triggered")
            evaluator.fire(alerts)
    except Exception as e:
        print(f"[alerts] Evaluation error: {e}")


# ── Enemy Scan ─────────────────────────────────────────────────────────────────
def run_enemy_scan(building_num: int, status_cb=None) -> str:
    """Scan enemy cars in a building.

    Flow mirrors team scan: click building → lobby appears → click ATTACK
    (instead of DEFEND) → same slot cycling via Next button → save screenshots.
    """
    cfg = reload_config()
    win = detect_window()
    get_tpl().refresh(win)

    ctx = ScanContext.create(win, cfg, status_cb)

    bldg_cfg = cfg.get("enemy_buildings", {}).get(str(building_num))
    if bldg_cfg is None:
        return (f"[ERROR] No enemy position for building {building_num}.\n"
                f"Add \"enemy_buildings\": {{\"{building_num}\": [x, y]}} to config.json")

    ctx.status(f"Enemy building {building_num} — entering…")
    pos   = win.sp(int(bldg_cfg[0]), int(bldg_cfg[1]))
    empty = not _enter_building(win, ctx, pos, _ATTACK_BTN_BL)
    if empty:
        return f"Enemy building {building_num}: empty — nothing to scan."

    opp_shot = pyautogui.screenshot(region=win.hud("opponent"))
    opp_shot.save(os.path.join(IMAGES_DIR, "debug_opponent_raw.PNG"))
    from bot.ocr import _prepare_for_ocr, _NAME_OCR_CJK, _resolve_cjk
    opp_proc = _prepare_for_ocr(opp_shot, convert_dark_text, upscale=2, minfilter=3)
    opp_proc.save(os.path.join(IMAGES_DIR, "debug_opponent_conv.PNG"))
    opponent_raw = _ocr(opp_proc, config='--oem 1 --psm 7').strip()
    if not opponent_raw:
        opponent_raw = _ocr(opp_proc, config='--oem 1 --psm 8').strip()
    # CJK fallback — if ASCII OCR returned nothing useful
    if not opponent_raw or len(re.sub(r'[^A-Za-z0-9]', '', opponent_raw)) < 2:
        cjk_raw = _ocr(opp_proc, config=_NAME_OCR_CJK).strip()
        if cjk_raw:
            resolved = _resolve_cjk(cjk_raw)
            print(f"[opponent] CJK: raw={cjk_raw!r} → {resolved!r}")
            if resolved:
                opponent_raw = resolved
    # EasyOCR fallback
    if not opponent_raw or len(re.sub(r'[^A-Za-z0-9]', '', opponent_raw)) < 2:
        try:
            from bot.easyocr_engine import read_name as _easy_read_name
            use_cjk = cfg.get("cjk_names", False)
            easy_opp, easy_conf = _easy_read_name(opp_shot, cjk=use_cjk)
            if easy_opp and easy_conf > 0.3:
                print(f"[opponent] EasyOCR: '{easy_opp}' (conf={easy_conf:.2f})")
                opponent_raw = easy_opp
        except Exception:
            pass
    opponent = re.sub(r'[^A-Za-z0-9_\- ]', '', opponent_raw).strip() or "enemy"
    ctx.status(f"  Opponent: {opponent}")

    build_r   = win.hud("build")
    captured  = 0
    max_slots = cfg.get("max_enemy_slots", 14)

    # War tracking: find or create war for this opponent
    war_id = None
    try:
        from bot.history import init_db, get_or_create_war, save_enemy_slot
        init_db()
        war_id = get_or_create_war(opponent)
    except Exception as e:
        print(f"[history] War tracking init error: {e}")

    def _capture(slot):
        """Capture one enemy slot. Returns player name for wrap detection."""
        nonlocal captured
        player_raw = _ocr_enemy_name(win)
        player     = re.sub(r'[^A-Za-z0-9_\- ]', '', player_raw).strip() or "unknown"
        player_dir = os.path.join(SCANS_DIR, opponent, player)
        os.makedirs(player_dir, exist_ok=True)
        idx  = len(os.listdir(player_dir)) + 1
        path = os.path.join(player_dir, f"{idx}.PNG")
        pyautogui.screenshot(region=build_r).save(path)
        captured += 1
        ctx.status(f"  Captured: {opponent}/{player}/{idx}.PNG")

        # Save to opponent_players DB
        if war_id is not None:
            try:
                save_enemy_slot(war_id, opponent, player,
                                building_num, screenshot=path)
            except Exception as e:
                print(f"[history] Enemy slot save error: {e}")
        return player

    _cycle_slots(win, ctx, _capture, _advance_slot,
                 build_r, check_lobby=True, max_slots=max_slots)
    _move_lobby(win, ctx)

    result = (f"Enemy building {building_num}: {captured} slot(s) captured.\n"
              f"Saved → {os.path.join(SCANS_DIR, opponent)}")
    ctx.status(result)
    return result
