"""bot/templates.py — Template image cache, HUD coord map, and navigation baselines.

All HUD coordinates and building positions are stored here as 1920×1080 baselines.
Templates auto-rescale when the window scale changes.
"""

import os
import pyautogui
from PIL import Image as Img

from bot.config import IMAGES_DIR, CFG, get_logger

_log = get_logger("templates")


# ── HUD layout (1920×1080 baseline — fixed game-UI positions) ─────────────────
_HUD_DEFAULTS = {
    "timer":           (793,  122, 258,  60),
    "opponent":        (1227, 135, 259,  53),
    "max_points":      (818,  210, 310,  92),
    "opp_points":      (1150, 215, 185,  65),
    "opp_bonus":       (1320, 185, 175, 110),
    "team_points":     (565,  213, 195,  77),
    "team_bonus":      (420,  170, 175, 110),
    "build":           (129,  326, 416, 407),  # car/build card screenshot area
    "enemy_name":      (180,  345, 260,  65),  # enemy car player-name text
    # Per-slot stat regions — verify with debug_slot_hp_raw.PNG / debug_slot_atk_raw.PNG
    "slot_hp":         (178,  685, 205,  50),  # ♥ HP value (after heart, fits 7 digits)
    "slot_atk":        (392,  685, 210,  50),  # ⚔ Attack value (after sword, fits 7 digits)
    "slot_status_px":  (228,  306,   1,   1),  # 1-px sample: green=DEF, red=ATK
}

# Apply any user overrides from config.json "hud_overrides"
_HUD = dict(_HUD_DEFAULTS)
for _k, _v in CFG.get("hud_overrides", {}).items():
    if _k in _HUD and len(_v) == 4:
        _HUD[_k] = tuple(_v)


def reload_hud_overrides():
    """Re-apply HUD overrides from CFG after config reload."""
    _HUD.clear()
    _HUD.update(_HUD_DEFAULTS)
    for k, v in CFG.get("hud_overrides", {}).items():
        if k in _HUD and len(v) == 4:
            _HUD[k] = tuple(v)


# Team building map-icon positions (1920×1080 baseline)
_TEAM_BUILDINGS_BL = {
    1: ( 438, 534),
    2: ( 349, 772),
    3: ( 739, 912),
    4: (1174, 930),
    5: (1498, 778),
    6: (1473, 507),
}

# Navigation button baselines (1920×1080)
_ESCAPE_BL     = (118, 1004)   # back/exit button
_DEFEND_BTN_BL = (716,  447)   # DEFEND octagon on lobby screen (team entry)
_ATTACK_BTN_BL = (1204, 447)   # ATTACK octagon on lobby screen (enemy entry)
_NEXT_BTN_BL   = (480,  843)   # Next slot button centre


# ── Template image cache ───────────────────────────────────────────────────────
class Templates:
    """Lazy-loading, auto-rescaling template image cache.

    Templates are rescaled whenever the Window scale changes so that
    pyautogui.locateOnScreen confidence comparisons remain valid.
    """

    _NAMES = ("lobby.PNG", "team_logo.png")

    def __init__(self):
        self._cache = {}
        self._scale = (None, None)

    def _load(self, filename: str, sx: float, sy: float):
        path = os.path.join(IMAGES_DIR, filename)
        if not os.path.isfile(path):
            _log.warning("Missing template: %s", filename)
            self._cache[filename] = None
            return
        img = Img.open(path).convert("RGB")
        if sx != 1.0 or sy != 1.0:
            nw = max(1, round(img.width  * sx))
            nh = max(1, round(img.height * sy))
            img = img.resize((nw, nh), Img.LANCZOS)
        self._cache[filename] = img

    def refresh(self, win):
        """Reload all templates scaled to the current window if scale changed."""
        if (win.sx, win.sy) == self._scale:
            return
        self._scale = (win.sx, win.sy)
        self._cache.clear()
        for name in self._NAMES:
            self._load(name, win.sx, win.sy)

    def get(self, filename: str):
        """Return the cached PIL Image, or None if the file was missing."""
        return self._cache.get(filename)

    def find(self, filename: str, region=None, confidence: float = 0.8):
        """Return pyautogui Box or None."""
        tpl = self.get(filename)
        if tpl is None:
            return None
        try:
            return pyautogui.locateOnScreen(tpl, region=region, confidence=confidence)
        except Exception as e:
            _log.error("locateOnScreen failed for %s: %s", filename, e)
            return None

    def find_center(self, filename: str, region=None, confidence: float = 0.8):
        """Return (cx, cy) or None."""
        box = self.find(filename, region=region, confidence=confidence)
        return pyautogui.center(box) if box else None


_TPL = Templates()


def get_tpl() -> Templates:
    """Return the module-level Templates singleton."""
    return _TPL
