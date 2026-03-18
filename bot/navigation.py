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

from bot.templates import (
    get_tpl, _ESCAPE_BL, _DEFEND_BTN_BL, _ATTACK_BTN_BL, _NEXT_BTN_BL
)


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
PIXEL_DIFF_THRESHOLD    = 12     # mean pixel diff below which = same frame (wrap)
SLOT_SETTLE_TIME        = 0.15   # seconds to wait for new slot to render
SCREEN_TRANSITION_TIMEOUT = 8.0  # seconds max for lobby transitions
LOGO_CACHE_MARGIN_PX    = 20     # pixels around cached logo position for fast search


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
        if (r, g, b) in {(255, 101, 100), (94, 106, 131)}:
            return False
        while (r, g, b) in {(144, 179, 235), (0, 0, 0)} and not _esc():
            pyautogui.click(x, y)
            time.sleep(0.05)
            r, g, b = pyautogui.screenshot(region=(x, y, 1, 1)).getpixel((0, 0))
        _move_enter(win, ctx, btn_bl)
        return not _esc()
    return False


def _advance_slot(win, build_region) -> bool:
    """Click the Next button and wait for the build card to change.

    Used by both team and enemy slot cycling — same Next button in both views.
    Polls a 40×40 center crop of the build region for change detection —
    much cheaper than screenshotting the full region on every 40ms poll.
    Returns True if the slot advanced, False if Next button not found.
    """
    # Compute a small center crop for fast change detection
    bx, by, bw, bh = build_region
    cx = bx + (bw - 40) // 2
    cy = by + (bh - 40) // 2
    poll_region = (cx, cy, 40, 40)

    pre = np.asarray(pyautogui.screenshot(region=poll_region))

    btn = None
    for attempt in range(NEXT_BTN_MAX_ATTEMPTS):
        btn = _find_next_btn_pos(win)
        if btn:
            break
        time.sleep(0.05)

    if btn is None:
        print(f"[advance] Next button NOT found after {NEXT_BTN_MAX_ATTEMPTS} attempts")
        return False

    print(f"[advance] Next button found at {btn} (attempt {attempt+1})")
    pyautogui.click(*btn)

    for i in range(ADVANCE_MAX_POLLS):
        time.sleep(ADVANCE_POLL_INTERVAL)
        if not np.array_equal(pre, np.asarray(pyautogui.screenshot(region=poll_region))):
            return True
    print(f"[advance] Frame did not change after clicking Next at {btn}")
    return False


# ── Unified slot cycling ───────────────────────────────────────────────────────
def _cycle_slots(win, ctx, on_slot, advance_fn, build_region,
                 check_lobby: bool = False, max_slots: int = 50) -> int:
    """Unified slot cycling used by both team and enemy scans.

    Args:
        on_slot(slot_index) -> str|None:  called once per slot; should return
                                          an identifier (e.g. player name) for
                                          semantic wrap detection. None = unknown.
        advance_fn(win, build_region) -> bool:  click next/select, True if slot changed
        build_region:   (x,y,w,h) car-card area — used for pixel-diff fallback
        check_lobby:    throttled lobby-visible check + _move_garage on exit (team only)
        max_slots:      hard upper bound

    End conditions (first matched wins):
        1. ESC held
        2. Lobby visible (check_lobby only)
        3. advance_fn returns False after retry
        4. Semantic wrap: same player name as slot 1 seen again
        5. Pixel-diff fallback: frame nearly identical to slot 1 (diff < 8)
        6. max_slots reached

    Returns total slots visited.
    """
    # Capture reference frame AFTER a brief settle — the building entry
    # animation may still be playing.
    time.sleep(SLOT_SETTLE_TIME)
    first_frame  = None     # re-captured after on_slot(1) for accuracy
    first_id     = None     # identifier from slot 1 (semantic wrap detection)
    slot         = 1
    _lobby_t     = time.time()  # grace period: skip lobby check on first iteration
    exit_reason  = "max_slots"

    while not _esc() and slot <= max_slots:
        if check_lobby and slot >= 3:
            # Only start lobby checks from slot 3 onward — we just entered
            # the building, so lobby can't be visible on the first few slots.
            now = time.time()
            if now - _lobby_t > LOBBY_THROTTLE_SEC:
                if _lobby_visible(win, ctx, LOBBY_HIGH_CONFIDENCE):
                    exit_reason = f"lobby visible at slot {slot}"
                    break
                _lobby_t = now

        slot_id = on_slot(slot)

        # Capture reference frame AFTER on_slot(1) processes — this ensures
        # the reference is the actual rendered slot 1, not a transition frame.
        # on_slot may move the mouse/take screenshots, so re-capture cleanly.
        if slot == 1:
            time.sleep(0.05)
            first_frame = np.asarray(pyautogui.screenshot(region=build_region))

        # Track first slot's identifier for semantic wrap detection.
        if slot == 1 and slot_id:
            first_id = slot_id
            print(f"[slots] slot 1 id={first_id!r}")

        # Semantic wrap: same player name as slot 1 — confirm with pixel diff.
        # A player can have up to 3 cars in one building, so same name alone
        # is NOT a wrap. Same name + same build-card frame = same car = wrap.
        if slot >= 2 and first_id and slot_id == first_id and first_frame is not None:
            cur = np.asarray(pyautogui.screenshot(region=build_region))
            diff = np.mean(np.abs(
                cur.astype(np.int32) - first_frame.astype(np.int32)))
            if diff < PIXEL_DIFF_THRESHOLD:
                exit_reason = (f"semantic+pixel wrap at slot {slot} "
                               f"(id={slot_id!r}, diff={diff:.1f})")
                print(f"[slots] {exit_reason}")
                break
            print(f"[slots] slot {slot} id={slot_id!r} matches slot 1 "
                  f"but diff={diff:.1f} — different car, continuing")

        print(f"[slots] slot {slot} processed (id={slot_id!r}), advancing…")

        advanced = advance_fn(win, build_region)
        if not advanced:
            # Retry twice with increasing pauses — transient Next-button failure
            for retry_delay in (0.3, 0.5):
                time.sleep(retry_delay)
                advanced = advance_fn(win, build_region)
                if advanced:
                    break
            if not advanced:
                exit_reason = f"no Next button at slot {slot}"
                print(f"[slots] Last slot: {slot} ({exit_reason})")
                break

        # ── Wrap detection ────────────────────────────────────────────────
        # Wait briefly for the new slot to settle, then peek at identity
        time.sleep(SLOT_SETTLE_TIME)

        # Pixel-diff fallback: frame nearly identical to slot 1 = wrap.
        # Skip if first_frame wasn't captured (shouldn't happen but be safe).
        if first_frame is None:
            slot += 1
            continue

        cur = np.asarray(pyautogui.screenshot(region=build_region))
        diff = np.mean(np.abs(cur.astype(np.int32) - first_frame.astype(np.int32)))
        print(f"[slots] slot {slot}→{slot+1} diff={diff:.1f}")

        if diff < PIXEL_DIFF_THRESHOLD:
            # Confirm with a second screenshot — frame already settled from
            # the first SLOT_SETTLE_TIME, so a brief pause suffices
            time.sleep(0.05)
            cur2 = np.asarray(pyautogui.screenshot(region=build_region))
            diff2 = np.mean(np.abs(
                cur2.astype(np.int32) - first_frame.astype(np.int32)))
            print(f"[slots] wrap confirm: diff2={diff2:.1f}")
            if diff2 < PIXEL_DIFF_THRESHOLD:
                exit_reason = f"pixel wrap at slot {slot} (diff={diff2:.1f})"
                print(f"[slots] Wrap detected: {exit_reason}")
                break
            print(f"[slots] false wrap avoided (diff {diff:.1f}→{diff2:.1f})")

        slot += 1

    if _esc():
        exit_reason = "ESC pressed"

    if check_lobby:
        _move_garage(win, ctx)   # safe to call even if already at lobby

    print(f"[slots] Building complete: {slot} slots visited ({exit_reason})")
    return slot
