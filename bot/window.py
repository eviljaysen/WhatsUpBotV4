"""bot/window.py — Game window detection and scaled coordinate helpers.

All baseline coordinates are 1920×1080. Use win.sp() / win.sr() / win.hud()
to convert to live screen coordinates. Never hardcode live pixel values.
"""

import ctypes
import ctypes.wintypes
import os

import pyautogui

from bot.config import (
    GAME_WINDOW_TITLE, CONTENT_OFFSET_X, CONTENT_OFFSET_Y, IMAGES_DIR
)
from bot.templates import _HUD


class Window:
    """Live game-canvas geometry with scale helpers.

    All baseline coordinates are 1920×1080. sp() / sr() scale them to the
    actual live window size, offsetting by the canvas origin (x, y).
    """

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x  = x
        self.y  = y
        self.w  = w
        self.h  = h
        self.sx = w / 1920
        self.sy = h / 1080

    def sp(self, bx: float, by: float):
        """Scale a 1920×1080 point to live screen coordinates."""
        return (self.x + round(bx * self.sx),
                self.y + round(by * self.sy))

    def sr(self, bx: float, by: float, bw: float, bh: float):
        """Scale a 1920×1080 region to a live (x, y, w, h) tuple."""
        return (self.x + round(bx * self.sx),
                self.y + round(by * self.sy),
                round(bw * self.sx),
                round(bh * self.sy))

    def hud(self, key: str):
        """Scale a named HUD region."""
        return self.sr(*_HUD[key])

    @property
    def region(self):
        return (self.x, self.y, self.w, self.h)

    def __repr__(self):
        return (f"Window({self.x},{self.y} {self.w}×{self.h} "
                f"scale=({self.sx:.3f},{self.sy:.3f}))")


def detect_window() -> Window:
    """Detect the game window and return a Window object.

    Tries to find the largest child render surface of the matching top-level
    window. Falls back to the parent client area, then to full screen.
    Saves a debug screenshot on each call.
    """
    u32  = ctypes.windll.user32
    PROC = ctypes.WINFUNCTYPE(ctypes.c_bool,
                               ctypes.wintypes.HWND,
                               ctypes.wintypes.LPARAM)

    top_wins = []

    def _top_cb(hwnd, _):
        if not u32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(u32.GetWindowTextLengthW(hwnd) + 1)
        u32.GetWindowTextW(hwnd, buf, len(buf))
        if GAME_WINDOW_TITLE.lower() not in buf.value.lower():
            return True
        cr = ctypes.wintypes.RECT()
        u32.GetClientRect(hwnd, ctypes.byref(cr))
        if cr.right > 400 and cr.bottom > 400:
            top_wins.append((cr.right * cr.bottom, hwnd, buf.value))
        return True

    u32.EnumWindows(PROC(_top_cb), 0)

    wx = wy = 0
    ww, wh = pyautogui.size()
    title  = "fullscreen"

    if top_wins:
        _, hwnd, title = max(top_wins)
        children = []

        def _child_cb(child, _):
            if not u32.IsWindowVisible(child):
                return True
            cr = ctypes.wintypes.RECT()
            u32.GetClientRect(child, ctypes.byref(cr))
            if cr.right > 400 and cr.bottom > 400:
                children.append((cr.right * cr.bottom, child, cr.right, cr.bottom))
            return True

        u32.EnumChildWindows(hwnd, PROC(_child_cb), 0)

        if children:
            _, child, cw, ch = max(children)
            pt = ctypes.wintypes.POINT(0, 0)
            u32.ClientToScreen(child, ctypes.byref(pt))
            wx, wy, ww, wh = pt.x, pt.y, cw, ch
        else:
            pt = ctypes.wintypes.POINT(0, 0)
            rc = ctypes.wintypes.RECT()
            u32.ClientToScreen(hwnd, ctypes.byref(pt))
            u32.GetClientRect(hwnd, ctypes.byref(rc))
            wx, wy, ww, wh = pt.x, pt.y, rc.right, rc.bottom
    else:
        print(f"[window] '{GAME_WINDOW_TITLE}' not found — full-screen fallback {ww}×{wh}")

    wx += CONTENT_OFFSET_X
    wy += CONTENT_OFFSET_Y
    ww -= CONTENT_OFFSET_X
    wh -= CONTENT_OFFSET_Y

    win = Window(wx, wy, ww, wh)
    print(f"  Window: '{title}'  ({wx},{wy})  {ww}×{wh}  scale=({win.sx:.4f},{win.sy:.4f})")

    try:
        pyautogui.screenshot(region=win.region).save(
            os.path.join(IMAGES_DIR, "debug_window.PNG"))
    except Exception as e:
        print(f"[window] Debug screenshot failed: {e}")

    return win
