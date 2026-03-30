"""bot/navigation.py — Mouse/keyboard navigation and slot cycling.

All game navigation logic lives here: entering buildings, advancing slots,
detecting lobby state, moving between screens. No OCR, no image processing.

pyautogui.PAUSE and MINIMUM_DURATION are set here — the zero values are
intentional performance optimizations. Do not change them.
"""

import time
import numpy as np

import pyautogui
import keyboard

from bot.config import get_logger
from bot.templates import (
    get_tpl, _ESCAPE_BL, _DEFEND_BTN_BL, _ATTACK_BTN_BL, _NEXT_BTN_BL
)

_log = get_logger("nav")


# ── pyautogui performance settings ────────────────────────────────────────────
pyautogui.useImageNotFoundException(False)
pyautogui.PAUSE           = 0   # remove default 0.1 s inter-call delay
pyautogui.MINIMUM_DURATION = 0  # remove mouse-move animation time

# ── Named constants ──────────────────────────────────────────────────────────
LOBBY_CENTRE_MARGIN     = 0.20   # fraction of window width trimmed each side
LOBBY_THROTTLE_SEC      = 0.5    # min interval between lobby-visible checks
LOBBY_HIGH_CONFIDENCE   = 0.9    # confidence for in-loop lobby checks
NEXT_BTN_SEARCH_MARGIN  = (60, 40)   # (half-width, half-height) around baseline
NEXT_BTN_WHITE_THRESH   = 240    # min channel value for "white" pixel
NEXT_BTN_MAX_ATTEMPTS   = 60     # poll attempts to find Next button
ADVANCE_MAX_POLLS       = 60     # poll attempts for frame change after click
ADVANCE_POLL_INTERVAL   = 0.04   # seconds between frame-change polls
WRAP_FP_THRESHOLD       = 5      # fingerprint diff below which = same car (wrap)

SLOT_SETTLE_TIME        = 0.15   # seconds to wait for new slot to render
SCREEN_TRANSITION_TIMEOUT = 8.0  # seconds max for lobby transitions
LOGO_CACHE_MARGIN_PX    = 20     # pixels around cached logo position for fast search

# Building icon pixel colours (team map screen)
_BLDG_EMPTY_COLOURS   = {(255, 101, 100), (94, 106, 131)}   # empty — skip
_BLDG_LOADING_COLOURS = {(144, 179, 235), (0, 0, 0)}        # loading — keep clicking


# ── Primitive checks ───────────────────────────────────────────────────────────
def _esc() -> bool:
    """Return True if the ESC key is currently held — used to abort scans."""
    return keyboard.is_pressed("esc")


def _lobby_visible(win, ctx, confidence: float = 0.8) -> bool:
    """Check if the lobby template is visible on screen.

    Searches the centre 60% of the window width — lobby screen never sits
    at the far edges — so ~40% fewer pixels for pyautogui to scan.
    Falls back to the full window region if the template is larger than
    the centre band (e.g. oversized lobby.PNG).
    """
    tpl = get_tpl().get("lobby.PNG")
    if tpl is None:
        return False
    margin = round(win.w * LOBBY_CENTRE_MARGIN)
    region = (win.x + margin, win.y, win.w - margin * 2, win.h)
    # Fall back to full window if template exceeds the centre-band region
    if tpl.width > region[2] or tpl.height > region[3]:
        region = win.region
    try:
        return pyautogui.locateOnScreen(
            tpl, confidence=confidence, region=region) is not None
    except ValueError:
        return False



# ── Next / Select button finders ───────────────────────────────────────────────
def _find_next_btn_pos(win):
    """Return the centroid of the white Next-button cluster, or None."""
    bx, by = _NEXT_BTN_BL
    mx, my = NEXT_BTN_SEARCH_MARGIN
    region = win.sr(bx - mx, by - my, mx * 2, my * 2)
    shot   = pyautogui.screenshot(region=region)
    arr    = np.asarray(shot, dtype=np.uint8)
    t      = NEXT_BTN_WHITE_THRESH
    white  = (arr[:, :, 0] > t) & (arr[:, :, 1] > t) & (arr[:, :, 2] > t)
    if not white.any():
        return None
    ys, xs = np.where(white)
    return (region[0] + int(xs.mean()), region[1] + int(ys.mean()))


# ── Screen transitions ─────────────────────────────────────────────────────────
def _move_garage(win, ctx):
    """Click the back button until the lobby template appears."""
    pos      = win.sp(*_ESCAPE_BL)
    deadline = time.time() + SCREEN_TRANSITION_TIMEOUT
    while not _lobby_visible(win, ctx) and not _esc() and time.time() < deadline:
        pyautogui.click(*pos)
        time.sleep(0.05)


def _move_lobby(win, ctx):
    """Click the back button until the lobby template disappears."""
    pos      = win.sp(*_ESCAPE_BL)
    deadline = time.time() + SCREEN_TRANSITION_TIMEOUT
    while _lobby_visible(win, ctx) and not _esc() and time.time() < deadline:
        pyautogui.click(*pos)
        time.sleep(0.05)


def _move_enter(win, ctx, btn_bl=_DEFEND_BTN_BL):
    """Click a lobby button (DEFEND or ATTACK) until the lobby template disappears."""
    pos      = win.sp(*btn_bl)
    deadline = time.time() + SCREEN_TRANSITION_TIMEOUT
    while _lobby_visible(win, ctx) and not _esc() and time.time() < deadline:
        pyautogui.click(*pos)
        time.sleep(0.05)


def _enter_building(win, ctx, pos, btn_bl=_DEFEND_BTN_BL) -> bool:
    """Click a building at pos, then click DEFEND or ATTACK to enter slot view.

    Pixel colour at the map icon determines building state:
      (255,101,100) or (94,106,131) = empty building — skip
      (144,179,235) or (0,0,0)     = loading/placeholder — keep clicking
    Returns True when inside the slot view, False to skip.
    """
    x, y = pos
    while not _esc():
        r, g, b = pyautogui.screenshot(region=(x, y, 1, 1)).getpixel((0, 0))
        pyautogui.moveTo(x, y)
        if (r, g, b) in _BLDG_EMPTY_COLOURS:
            return False
        while (r, g, b) in _BLDG_LOADING_COLOURS and not _esc():
            pyautogui.click(x, y)
            time.sleep(0.05)
            r, g, b = pyautogui.screenshot(region=(x, y, 1, 1)).getpixel((0, 0))
        _move_enter(win, ctx, btn_bl)
        return not _esc()
    return False


def _advance_slot(win, build_region) -> bool:
    """Click the Next button and wait for the build card to change.

    Used by both team and enemy slot cycling — same Next button in both views.
    Polls a 40×40 center crop for fast change detection, then verifies with
    a full-region check to filter out animation noise that only touches the center.
    Returns True if the slot advanced, False if Next button not found.
    """
    # Compute a small center crop for fast change detection
    bx, by, bw, bh = build_region
    cx = bx + (bw - 40) // 2
    cy = by + (bh - 40) // 2
    poll_region = (cx, cy, 40, 40)

    pre_crop = np.asarray(pyautogui.screenshot(region=poll_region))
    pre_full = np.asarray(pyautogui.screenshot(region=build_region))

    btn = None
    for attempt in range(NEXT_BTN_MAX_ATTEMPTS):
        btn = _find_next_btn_pos(win)
        if btn:
            break
        time.sleep(0.05)

    if btn is None:
        # Last-ditch: click the baseline Next button position directly
        fallback = win.sp(*_NEXT_BTN_BL)
        _log.warning("Next button NOT found — trying baseline click at %s", fallback)
        pyautogui.click(*fallback)
    else:
        _log.debug("Next button found at %s (attempt %d)", btn, attempt + 1)
        pyautogui.click(*btn)

    # Wait for center crop to change (fast check)
    center_changed = False
    for i in range(ADVANCE_MAX_POLLS):
        time.sleep(ADVANCE_POLL_INTERVAL)
        if not np.array_equal(pre_crop, np.asarray(pyautogui.screenshot(region=poll_region))):
            center_changed = True
            break

    if not center_changed:
        _log.warning("Frame did not change after clicking")
        return False

    # Verify full region actually changed — center crop can be tricked by
    # transient animation overlays that appear/disappear without advancing
    time.sleep(0.1)
    post_full = np.asarray(pyautogui.screenshot(region=build_region))
    full_diff = np.mean(np.abs(post_full.astype(np.int32) - pre_full.astype(np.int32)))
    if full_diff < 3.0:
        _log.warning("Center changed but full region diff=%.1f — "
                     "likely animation noise, not a real advance", full_diff)
        return False
    return True


# ── Unified slot cycling ───────────────────────────────────────────────────────
def _make_fingerprint(build_region) -> tuple:
    """Create a compact fingerprint of the car image for wrap comparison.

    Downscales to 32×32 grayscale — small enough for fast comparison,
    large enough to distinguish different cars reliably.

    Returns (fingerprint, raw_frame) — raw_frame reused for stuck detection
    to avoid a redundant screenshot.
    """
    import cv2
    shot = pyautogui.screenshot(region=build_region)
    raw = np.asarray(shot)
    gray = np.asarray(shot.convert('L'), dtype=np.uint8)
    fp = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return fp, raw


def _fingerprint_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two fingerprints (0–255 scale)."""
    return float(np.mean(np.abs(a.astype(np.int32) - b.astype(np.int32))))



def _cycle_slots(win, ctx, on_slot, advance_fn, build_region,
                 check_lobby: bool = False, max_slots: int = 50,
                 on_wrap=None) -> int:
    """Unified slot cycling used by both team and enemy scans.

    Args:
        on_slot(slot_index) -> dict|None:
            Called once per slot. Returns a dict with key:
                name: str       — player name (for wrap detection + logging)
            Returns None if the slot should be skipped (on_slot crashed).
        advance_fn(win, build_region) -> bool:
            Click next, True if slot changed.
        build_region:   (x,y,w,h) car-card area
        check_lobby:    throttled lobby-visible check (team only)
        max_slots:      hard upper bound
        on_wrap:        optional callback(slot_info) — called when wrap detected,
                        BEFORE breaking, to let the caller undo on_slot effects
                        (pop slot_results, decrement counts, delete screenshot)

    Wrap detection (both must match slot 1):
        1. Same player name
        2. Similar car screenshot (fingerprint diff < threshold)

    HP/ATK is NOT used for wrap detection — live OCR is too noisy
    (values differ by 50%+ between reads of the same slot due to CNN
    returning garbage like "2" or wildly different digits). The 32×32
    grayscale fingerprint reliably distinguishes different cars from
    the same player (diff 0–2 for same car vs 12–30+ for different cars).

    Other end conditions:
        - ESC held
        - Lobby visible (check_lobby only)
        - advance_fn returns False after retry
        - Stuck detection: frame unchanged after advance
        - max_slots reached

    Returns total slots visited.
    """
    time.sleep(SLOT_SETTLE_TIME)
    slot_1_name  = None    # name from on_slot(1)
    slot_1_fp    = None    # fingerprint of slot 1 car screenshot
    prev_frame   = None    # previous slot frame for stuck detection
    slot         = 1
    _lobby_t     = time.time()
    exit_reason  = "max_slots"

    while not _esc() and slot <= max_slots:
        if check_lobby and slot >= 3:
            now = time.time()
            if now - _lobby_t > LOBBY_THROTTLE_SEC:
                if _lobby_visible(win, ctx, LOBBY_HIGH_CONFIDENCE):
                    exit_reason = f"lobby visible at slot {slot}"
                    break
                _lobby_t = now

        # ── Process slot — protected from exceptions ───────────────────
        slot_info = None
        try:
            slot_info = on_slot(slot)
        except Exception as e:
            _log.error("on_slot(%d) CRASHED: %s", slot, e, exc_info=True)

        # Take fingerprint of current car screenshot (also returns raw frame
        # for stuck detection — avoids a redundant screenshot)
        cur_fp, prev_frame = _make_fingerprint(build_region)

        cur_name = slot_info['name'] if slot_info else None

        # Store slot 1 reference for wrap comparison
        if slot == 1 and cur_name:
            slot_1_name = cur_name
            slot_1_fp   = cur_fp
            _log.info("slot 1 ref: name=%r", cur_name)

        # ── Wrap detection (slot 2+) ───────────────────────────────────
        # A wrap is when we've cycled back to the EXACT same car as slot 1.
        # Both criteria must match:
        #   1. Same player name
        #   2. Similar car screenshot fingerprint (diff < threshold)
        if slot >= 2 and slot_1_name and cur_name == slot_1_name:
            fp_diff = _fingerprint_diff(cur_fp, slot_1_fp)
            _log.info("slot %d matches slot 1 name (%r), fingerprint diff=%.1f",
                      slot, cur_name, fp_diff)
            if fp_diff < WRAP_FP_THRESHOLD:
                exit_reason = (f"wrap at slot {slot}: "
                               f"{cur_name!r} fp_diff={fp_diff:.1f}")
                _log.info("%s", exit_reason)
                if on_wrap:
                    on_wrap(slot_info)
                slot -= 1   # don't count the wrap slot
                break
            else:
                _log.info("same name but different car (fp_diff=%.1f) "
                          "— continuing", fp_diff)

        if slot_info:
            _log.info("slot %d captured (name=%r), advancing…", slot, slot_info['name'])
        else:
            _log.info("slot %d (no info), advancing…", slot)

        advanced = advance_fn(win, build_region)
        if not advanced:
            for retry_delay in (0.3, 0.5, 0.8):
                time.sleep(retry_delay)
                advanced = advance_fn(win, build_region)
                if advanced:
                    break
            if not advanced:
                exit_reason = f"no Next button at slot {slot}"
                _log.info("Last slot: %d (%s)", slot, exit_reason)
                break

        # ── Post-advance checks ────────────────────────────────────────
        time.sleep(SLOT_SETTLE_TIME)

        # Stuck detection: frame unchanged = advance didn't work
        if prev_frame is not None:
            new_frame = np.asarray(pyautogui.screenshot(region=build_region))
            stuck_diff = np.mean(np.abs(
                new_frame.astype(np.int32) - prev_frame.astype(np.int32)))
            if stuck_diff < 3.0:
                _log.warning("Stuck! diff vs prev=%.1f — retrying advance",
                             stuck_diff)
                time.sleep(0.3)
                advanced = advance_fn(win, build_region)
                if not advanced:
                    exit_reason = f"stuck at slot {slot}"
                    _log.warning("Still stuck: %s", exit_reason)
                    break
                time.sleep(SLOT_SETTLE_TIME)

        slot += 1

    if _esc():
        exit_reason = "ESC pressed"

    if check_lobby:
        _move_garage(win, ctx)

    _log.info("Building complete: %d slots visited (%s)", slot, exit_reason)
    return slot
