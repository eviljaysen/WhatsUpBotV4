"""bot/ocr.py — All OCR operations.

Covers: Tesseract config strings, score/number OCR, player name pipeline
(ASCII multi-pass + EasyOCR fallback), NameMatcher class (replaces v3 global
_name_templates dict), and enemy-panel name OCR.

NameMatcher is instantiated per-scan inside ScanContext — no module-level
mutable state for template data.
"""

import os
import re
import difflib
import threading

import numpy as np
import cv2
import pytesseract
import pyautogui
from PIL import Image as Img, ImageFilter

from bot.config import (
    IMAGES_DIR, NAME_TMPL_DIR, CFG,
    _OCR_CORRECTIONS, _KNOWN_NAMES, _NAME_UPPER_MAP,
    get_logger,
)
from bot.vision import convert_to_bw

_log = get_logger("ocr")


# ── Tesseract setup ────────────────────────────────────────────────────────────
_TESS_EXE = CFG.get("tesseract_path", r'C:\Program Files\Tesseract-OCR\tesseract.exe')
if os.path.isfile(_TESS_EXE):
    pytesseract.pytesseract.tesseract_cmd = _TESS_EXE
    _log.info("Tesseract configured.")
else:
    _log.warning("Tesseract not found at default path — ensure it is on PATH.")


# ── OCR config strings ─────────────────────────────────────────────────────────
_BONUS_OCR     = r"--oem 1 --psm 7 -c tessedit_char_whitelist=+0123456789"
_NAME_OCR      = r"--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
_NAME_OCR_PSM8 = r"--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
_NUMBERS_OCR   = r"--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789"
_TIMER_OCR     = r"--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789hm "

# Template thumbnail dimensions
_TMPL_W, _TMPL_H = 160, 40


# ── Core OCR helpers ───────────────────────────────────────────────────────────
def _ocr(image, config: str) -> str:
    return pytesseract.image_to_string(image, config=config)


def _digits(text: str) -> int:
    d = re.sub(r'[^0-9]', '', text)
    return int(d) if d else 0


def _apply_correction(text: str) -> str:
    """Look up text in OCR corrections (exact then uppercased)."""
    return _OCR_CORRECTIONS.get(text, _OCR_CORRECTIONS.get(text.upper(), text))


def _prepare_for_ocr(img, converter, upscale: int = 1, pad: int = 10,
                     minfilter: int = 3) -> Img.Image:
    """Upscale → convert → MinFilter → pad white border."""
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Img.NEAREST)
    img = converter(img)
    if minfilter > 0:
        img = img.filter(ImageFilter.MinFilter(minfilter))
    arr = np.pad(np.asarray(img), pad, constant_values=255)
    return Img.fromarray(arr)


def locate_number(region, converter=None, ocr_config=_NUMBERS_OCR,
                  debug_name=None, upscale: int = 1, minfilter: int = 3,
                  field: str = '', valid_range: tuple = None,
                  use_cnn: bool = True) -> int:
    """Screenshot a region, OCR it, and return the integer value.

    Args:
        region: (x, y, w, h) live screen region
        converter: image converter fn (defaults to convert_dark_text)
        ocr_config: Tesseract config string
        debug_name: if set, saves raw + conv debug images to images/
        upscale: integer upscale factor before OCR
        minfilter: MinFilter kernel size (0 = skip)
        field: field type for CNN class masking (e.g. 'slot_hp', 'timer')
        valid_range: (min, max) — reject and don't save training if outside
        use_cnn: if False, skip CNN and go straight to Tesseract
    """
    from bot.vision import convert_dark_text  # avoid circular at module level
    shot = pyautogui.screenshot(region=region)
    if debug_name:
        shot.save(os.path.join(IMAGES_DIR, f"debug_{debug_name}_raw.PNG"))
    processed = _prepare_for_ocr(
        shot, converter if converter is not None else convert_dark_text,
        upscale, minfilter=minfilter)
    if debug_name:
        processed.save(os.path.join(IMAGES_DIR, f"debug_{debug_name}_conv.PNG"))

    result = 0
    label = debug_name or field or "ocr"

    def _in_range(v: int) -> bool:
        if not valid_range:
            return True
        return valid_range[0] <= v <= valid_range[1]

    # ── CNN — try first ───────────────────────────────────────────────────────
    try:
        from bot.ocr_model import is_model_ready, predict_text
        if use_cnn and is_model_ready():
            ml_text, ml_conf = predict_text(processed, field=field or debug_name or "")
            if ml_text and ml_conf >= 0.8:
                ml_val = _digits(ml_text)
                if ml_val > 0 and _in_range(ml_val):
                    _log.info("%s: CNN → '%s' (conf=%.2f) = %d",
                              label, ml_text, ml_conf, ml_val)
                    result = ml_val
                elif ml_val > 0:
                    _log.debug("%s: CNN → '%s' (conf=%.2f) = %d outside valid range "
                               "— falling through to Tesseract",
                               label, ml_text, ml_conf, ml_val)
    except Exception as e:
        _log.error("CNN error: %s", e)

    # ── Tesseract fallback ────────────────────────────────────────────────────
    if result == 0:
        raw_text = _ocr(processed, config=ocr_config).strip()
        tess_val = _digits(raw_text)
        if debug_name:
            _log.debug("%s: Tesseract raw='%s' → %d", label, raw_text, tess_val)
        if tess_val > 0 and _in_range(tess_val):
            result = tess_val
        elif tess_val > 0:
            _log.debug("%s: Tesseract value %d outside valid range — skipping",
                       label, tess_val)
        # Retry bonus with PSM 8 (single word) if PSM 7 failed
        if result == 0 and debug_name and "bonus" in debug_name:
            alt_config = ocr_config.replace("--psm 7", "--psm 8")
            alt_text = _ocr(processed, config=alt_config).strip()
            alt_val = _digits(alt_text)
            _log.debug("%s: retry PSM 8: raw='%s' → %d", label, alt_text, alt_val)
            if alt_val > 0 and _in_range(alt_val):
                result = alt_val

    # ── EasyOCR fallback — only when nothing else worked ─────────────────────
    if result == 0 and debug_name:
        try:
            from bot.easyocr_engine import read_number
            easy_val, easy_conf = read_number(processed)
            if easy_val > 0 and easy_conf > 0.3 and _in_range(easy_val):
                _log.info("%s: EasyOCR fallback: %d (conf=%.2f)", label, easy_val, easy_conf)
                result = easy_val
            elif easy_val > 0:
                _log.debug("%s: EasyOCR: %d (conf=%.2f) — skipped (out of range or low conf)",
                           label, easy_val, easy_conf)
        except Exception as e:
            _log.debug("%s: EasyOCR unavailable: %s", label, e)

    # ── Final warning if all engines failed valid_range ───────────────────────
    if result == 0 and valid_range:
        _log.debug("%s: all engines returned 0 or out-of-range", label)

    # Auto-save training sample when we got a valid result
    if result > 0 and debug_name:
        _save_ocr_training_sample(debug_name, shot, processed, result)
    return result


_train_save_lock = threading.Lock()


def _save_ocr_training_sample(field: str, raw_img, conv_img, value):
    """Save an OCR training sample (raw + conv image with label).

    Args:
        value: int or str — the ground-truth label for the image.
    """
    import time
    with _train_save_lock:
        train_dir = os.path.join(os.path.dirname(IMAGES_DIR), "training_data", "ocr", field)
        os.makedirs(train_dir, exist_ok=True)

        # Cap samples per field (names get more since there are many players)
        cap = 250 if field == "name" else 100
        existing = sorted(f for f in os.listdir(train_dir) if f.endswith("_raw.png"))
        if len(existing) >= cap:
            # Remove oldest pair
            oldest = existing[0].replace("_raw.png", "")
            for suffix in ("_raw.png", "_conv.png"):
                p = os.path.join(train_dir, oldest + suffix)
                if os.path.isfile(p):
                    os.remove(p)

        ts = int(time.time() * 1000)
        label = str(value)
        raw_img.save(os.path.join(train_dir, f"{ts}_{label}_raw.png"))
        conv_img.save(os.path.join(train_dir, f"{ts}_{label}_conv.png"))


def save_stat_correction_sample(field: str, raw_img, conv_img, value: int):
    """Persist a user-corrected stat value as an OCR training sample."""
    _save_ocr_training_sample(field, raw_img, conv_img, value)


def get_build_stat_images(image_path: str) -> tuple:
    """Extract HP and ATK region images from a build card for correction/training.

    Replicates the gap-based splitting used by ocr_build_stats to isolate the
    HP and ATK digit regions as PIL Images (raw crop + binarized conv).

    Returns:
        (hp_raw, hp_conv, atk_raw, atk_conv) — any element may be None on failure.
    """
    from PIL import Image as Img
    try:
        img = Img.open(image_path).convert("RGB")
    except Exception:
        return None, None, None, None

    w, h = img.size
    stat_top = int(h * 0.86)
    stat_raw = img.crop((0, stat_top, w, h))

    from bot.vision import erase_stat_icons
    stat_clean = erase_stat_icons(stat_raw)
    arr = np.asarray(stat_clean, dtype=np.uint8)

    gray = np.mean(arr, axis=2)
    bw = np.full(gray.shape, 255, dtype=np.uint8)
    bw[gray < 170] = 0
    col_dark = np.sum(bw == 0, axis=0)

    gaps = []
    gap_start = None
    for x in range(len(col_dark)):
        if col_dark[x] == 0:
            if gap_start is None:
                gap_start = x
        else:
            if gap_start is not None:
                gap_len = x - gap_start
                if gap_len >= 3 and gap_start > 10:
                    gaps.append((gap_start, x, gap_len))
                gap_start = None
    if gaps and gaps[-1][1] >= len(col_dark) - 2:
        gaps = gaps[:-1]
    if not gaps:
        return stat_raw, None, stat_raw, None

    widest_idx = max(range(len(gaps)), key=lambda i: gaps[i][2])
    hp_end     = gaps[widest_idx][0]
    widest_end = gaps[widest_idx][1]
    atk_start = widest_end
    search_end = min(widest_end + 55, len(col_dark))
    post_gap_start = None
    for _x in range(widest_end, search_end):
        if col_dark[_x] == 0:
            if post_gap_start is None:
                post_gap_start = _x
        else:
            if post_gap_start is not None:
                atk_start = _x
                break
    hp_start  = gaps[0][1] if gaps[0][2] >= 4 and gaps[0][0] < 55 else 25
    hp_start  = max(hp_start, 25)
    if widest_idx == 0:
        hp_start = 25
        hp_end   = gaps[0][0]

    try:
        hp_raw   = stat_raw.crop((hp_start,  0, hp_end,         stat_raw.height))
        atk_raw  = stat_raw.crop((atk_start, 0, stat_raw.width, stat_raw.height))
        hp_conv  = Img.fromarray(bw[:, hp_start:hp_end])
        atk_conv = Img.fromarray(bw[:, atk_start:])
    except Exception:
        return stat_raw, None, stat_raw, None

    return hp_raw, hp_conv, atk_raw, atk_conv


# ── Name OCR helpers ───────────────────────────────────────────────────────────
def _prep_name_image(shot) -> Img.Image:
    """3× upscale → B&W → MinFilter(3) → 10px white pad.

    3× (vs 2×) gives Tesseract more pixels on the typically small name region.
    MinFilter thickens thin strokes and suppresses isolated noise pixels.
    Padding prevents Tesseract from clipping characters at the image edge.
    """
    bw  = convert_to_bw(shot)
    bw3 = bw.resize((bw.width * 3, bw.height * 3), Img.NEAREST)
    bw3 = bw3.filter(ImageFilter.MinFilter(3))
    arr = np.pad(np.asarray(bw3), 10, constant_values=255)
    return Img.fromarray(arr)


def _read_name(proc, cfg: str) -> str:
    """OCR proc with cfg, normalise to ASCII, apply corrections."""
    t = _ocr(proc, config=cfg).strip()
    t = ' '.join(t.split()).encode('ascii', 'ignore').decode('ascii')
    return _apply_correction(t.upper())


def _name_match(text: str) -> list:
    return difflib.get_close_matches(
        text.upper(), list(_NAME_UPPER_MAP), n=1, cutoff=CFG.get("name_match_cutoff", 0.80))



# ── NameMatcher — per-scan template matching ───────────────────────────────────
class NameMatcher:
    """Manages name-region templates for CJK/unreadable player names.

    Per-scan instance (stored in ScanContext). Templates loaded from disk at
    init. Matches using Jaccard IoU on ink pixels — ignores white background.
    Capped at 5 templates per player (oldest dropped first).

    Uses a class-level cache: templates are only reloaded from disk when
    the name_templates directory mtime changes (new files added/removed).
    """

    _W, _H = _TMPL_W, _TMPL_H
    _cached_templates: dict = {}   # shared across instances
    _cached_mtime: float = 0.0     # mtime of NAME_TMPL_DIR when last loaded

    def __init__(self):
        self._templates: dict = {}  # player → list of (uint8_arr, ink_mask)
        self._load_all()

    def _load_all(self):
        if not os.path.isdir(NAME_TMPL_DIR):
            return
        try:
            dir_mtime = os.path.getmtime(NAME_TMPL_DIR)
        except OSError:
            dir_mtime = 0.0

        # Reuse cached templates if directory hasn't changed
        if (dir_mtime == NameMatcher._cached_mtime
                and NameMatcher._cached_mtime > 0
                and NameMatcher._cached_templates):
            # Deep copy: each instance gets its own mutable lists
            self._templates = {
                p: list(entries)
                for p, entries in NameMatcher._cached_templates.items()
            }
            _log.debug("name_matcher: reused cached templates (%d total)",
                      sum(len(v) for v in self._templates.values()))
            return

        for fname in sorted(os.listdir(NAME_TMPL_DIR)):
            if not fname.endswith(".png"):
                continue
            player = fname.rsplit("_", 1)[0]
            path   = os.path.join(NAME_TMPL_DIR, fname)
            try:
                arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if arr is not None:
                    self._templates.setdefault(player, []).append((arr, arr < 128))
            except Exception as e:
                _log.error("name_matcher: failed to load %s: %s", fname, e)
        # Cap at 5 templates per player (sorted order = oldest first)
        for player in self._templates:
            if len(self._templates[player]) > 5:
                self._templates[player] = self._templates[player][-5:]
        # Update class-level cache
        NameMatcher._cached_templates = {
            p: list(entries) for p, entries in self._templates.items()
        }
        NameMatcher._cached_mtime = dir_mtime
        _log.info("name_matcher: loaded %d templates from disk",
                  sum(len(v) for v in self._templates.values()))

    @staticmethod
    def _binarise(shot) -> np.ndarray:
        """Return a binary thumbnail of a name-region shot."""
        gray = np.asarray(shot.convert('L'), dtype=np.uint8)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_img = Img.fromarray(bw)
        bw_img = bw_img.resize((bw_img.width * 3, bw_img.height * 3), Img.NEAREST)
        bw_img = bw_img.filter(ImageFilter.MinFilter(3))
        arr    = np.asarray(bw_img, dtype=np.uint8)
        return cv2.resize(arr, (_TMPL_W, _TMPL_H), interpolation=cv2.INTER_AREA)

    def match(self, shot) -> str:
        """Compare shot against all saved templates. Return player name or ''.

        Uses Jaccard IoU on ink pixels only. Background-dominated pixel
        equality is not used — different names score ~0, same name scores 0.4–0.9.
        """
        if not self._templates:
            return ''
        arr     = self._binarise(shot)
        ink_arr = arr < 128
        best_score, best_player = 0.0, ''
        for player, templates in self._templates.items():
            for tpl, ink_tpl in templates:
                union = int(np.sum(ink_arr | ink_tpl))
                if union == 0:
                    continue
                score = float(np.sum(ink_arr & ink_tpl)) / union
                if score > best_score:
                    best_score, best_player = score, player
        if best_score >= CFG.get("tmpl_threshold", 0.40):
            _log.info("name_matcher: matched %r (score=%.3f)", best_player, best_score)
            return best_player
        return ''

    def save_template(self, player: str, shot):
        """Save a name-region shot as a template for player. Cap at 5."""
        if not player:
            return
        arr     = self._binarise(shot)
        entries = self._templates.setdefault(player, [])
        idx     = len(entries) + 1
        entries.append((arr, arr < 128))
        if len(entries) > 5:
            entries.pop(0)
        path = os.path.join(NAME_TMPL_DIR, f"{player}_{idx:02d}.png")
        cv2.imwrite(path, arr)
        # Invalidate class-level cache so next NameMatcher picks up the new file
        NameMatcher._cached_mtime = 0.0
        _log.info("name_matcher: saved template: %s", path)


# ── Player name OCR — full pipeline ───────────────────────────────────────────
def ocr_player_name(card, ctx) -> str:
    """OCR the player name using the card's detected name region.

    Pipeline (stops at first confident match):
    1. Template match (IoU) — fast, no Tesseract/ML needed
    2. ML model prediction (fast ~5ms, most accurate when trained)
    3. Multi-pass ASCII: PSM 7, then PSM 8
       - Each pass also tries stripping 1 leading char (logo-edge bleed)
    4. EasyOCR fallback
    5. Low-cutoff fuzzy match (0.60)
    6. Correction dialog (if correction_cb provided)

    If all passes fail, re-captures the name region after 200ms and retries
    once — the first capture may have hit a transition frame.

    Args:
        card: CardInfo with name_region and paw_end
        ctx: ScanContext (uses name_matcher, last_name_shot, correction_* fields)
    """
    import time as _time
    result = _ocr_player_name_once(card, ctx)

    # Check if result is a known player (matched against config)
    known = bool(result) and result in _KNOWN_NAMES
    if known:
        return result

    # Retry once — re-capture after a brief settle in case of transition frame
    _log.info("name: first attempt returned %r (unmatched) — retrying after 200ms",
              result)
    _time.sleep(0.2)
    result2 = _ocr_player_name_once(card, ctx)
    known2 = bool(result2) and result2 in _KNOWN_NAMES
    if known2:
        _log.info("name: retry succeeded: %r", result2)
        return result2

    # Use whichever raw text is longer (more likely to be real)
    raw = result2 if len(result2.strip()) > len(result.strip()) else result

    # Last resort: correction dialog (blocks scan thread up to 30s)
    if len(raw.strip()) >= 1 and ctx.correction_cb is not None:
        ctx.correction_event.clear()
        ctx.correction_result[0] = raw
        ctx.correction_cb(raw)
        ctx.correction_event.wait(timeout=30)
        player = ctx.correction_result[0] or raw
        matches = _name_match(player)
        if matches:
            player = _NAME_UPPER_MAP[matches[0]]
        _log.info("name: correction dialog → %r", player)
        return player

    return raw


def _ocr_player_name_once(card, ctx) -> str:
    """Single attempt at OCR'ing the player name. See ocr_player_name."""
    shot = pyautogui.screenshot(region=card.name_region)

    # Blank paw icon pixels — use 160 (mid-gray), NOT 255.
    # Pure white triggers convert_to_bw → black ink → spurious OCR chars.
    if card.paw_end > 0:
        arr = np.asarray(shot, dtype=np.uint8).copy()
        arr[:, :card.paw_end] = 160
        shot = Img.fromarray(arr)

    shot.save(os.path.join(IMAGES_DIR, "debug_name_raw.PNG"))
    ctx.last_name_shot[0] = shot

    # Pass 1: Template match (fast, no Tesseract)
    tmpl = ctx.name_matcher.match(shot)
    if tmpl:
        _log.info("name: template → %r", tmpl)
        return tmpl

    # Pass 2: ML model (fast ~5ms, most accurate when trained)
    if ctx.ml_model_ready is None:
        try:
            from bot.name_model import is_model_ready
            ctx.ml_model_ready = is_model_ready()
        except Exception as e:
            _log.error("name: ML model check failed: %s", e)
            ctx.ml_model_ready = False
    if ctx.ml_model_ready:
        from bot.name_model import predict_name
        ml_name, ml_conf = predict_name(shot)
        ml_threshold = CFG.get("ml_name_confidence", 0.85)
        if ml_name and ml_conf >= ml_threshold:
            _log.info("name: ML model → %r (conf=%.3f)", ml_name, ml_conf)
            proc = _prep_name_image(shot)
            _save_ocr_training_sample("name", shot, proc, ml_name)
            return ml_name
        elif ml_name:
            _log.debug("name: ML model low-conf: %r (conf=%.3f < %.2f)",
                       ml_name, ml_conf, ml_threshold)

    proc = _prep_name_image(shot)
    proc.save(os.path.join(IMAGES_DIR, "debug_name_conv.PNG"))

    # Pass 3: Multi-pass ASCII OCR (slower, handles unseen names)
    raw = player = ''
    matches = []
    # Only save a name template when ASCII OCR fails and CJK/EasyOCR is needed.
    # ASCII-readable names (ROBINHOOD, PATRICK, …) must NOT get templates — the
    # IoU matcher is not discriminative enough at 160×40 for Latin characters and
    # will false-match across players, causing immediate wrap detection failure.
    _save_tmpl = False
    for cfg_str in (_NAME_OCR, _NAME_OCR_PSM8):
        candidate = _read_name(proc, cfg_str)
        if not raw:
            raw = candidate
        variants = [candidate]
        if len(candidate) > 3:
            variants.append(candidate[1:])
        for text in variants:
            m = _name_match(text)
            if m:
                raw, matches = text, m
                break
        if matches:
            break

    # EasyOCR fallback — runs on the RAW shot (EasyOCR has its own preprocessing)
    if not matches:
        try:
            from bot.easyocr_engine import read_name as _easy_read_name
            easy_text, easy_conf = _easy_read_name(shot, cjk=False)
            if easy_text and easy_conf > 0.15:
                _log.info("name: EasyOCR: '%s' (conf=%.2f)", easy_text, easy_conf)
                # Try exact correction first
                corrected_easy = _OCR_CORRECTIONS.get(
                    easy_text, _OCR_CORRECTIONS.get(easy_text.upper(), ''))
                if corrected_easy:
                    m = _name_match(corrected_easy)
                    if m:
                        raw, matches = corrected_easy, m
                        _save_tmpl = True
                # Direct fuzzy match against player names
                if not matches:
                    m = _name_match(easy_text)
                    if m:
                        raw, matches = easy_text, m
                        _save_tmpl = True
            elif easy_text:
                _log.debug("name: EasyOCR low-conf: '%s' (conf=%.2f)", easy_text, easy_conf)
        except Exception as e:
            _log.debug("name: EasyOCR unavailable: %s", e)

    # Pre-dialog correction check
    if not matches:
        corrected = _OCR_CORRECTIONS.get(raw, _OCR_CORRECTIONS.get(raw.upper(), ''))
        if corrected:
            m = _name_match(corrected)
            if m:
                raw, matches = corrected, m

    # Last-resort fuzzy match with lower cutoff (0.60) — catches partial OCR
    # results like "ROBINHOO" → "ROBINHOOD" or "MUSHI" → "MUSCHI"
    if not matches and len(raw.strip()) >= 3:
        m = difflib.get_close_matches(
            raw.upper(), list(_NAME_UPPER_MAP), n=1, cutoff=0.60)
        if m:
            _log.info("name: low-cutoff fuzzy match: %r → %r", raw, m[0])
            matches = m

    if matches:
        player = _NAME_UPPER_MAP[matches[0]]
        _log.info("name: raw=%r matched=%r", raw, player)
        # Save OCR training sample for the character model
        if proc is not None:
            _save_ocr_training_sample("name", shot, proc, player)
        # Save template only for CJK/EasyOCR-resolved names — ASCII names must not
        # get templates since IoU at 160×40 cannot distinguish Latin characters.
        if _save_tmpl:
            ctx.name_matcher.save_template(player, shot)
        return player

    # No match found — return raw text (caller may retry or show dialog)
    _log.warning("name: raw=%r no match found", raw)
    return raw


def ocr_build_stats(image_path: str) -> tuple:
    """Extract HP and ATK values from a saved build card image.

    The build card has a stat line at the bottom: ♥ HP_VALUE ⚔ ATK_VALUE
    Uses gap-based splitting: finds the widest gap between dark pixel groups
    to separate the heart+HP section from the sword+ATK section.

    Args:
        image_path: path to saved build PNG

    Returns:
        (hp: int, atk: int) — (0, 0) on failure
    """
    from PIL import Image as Img
    try:
        img = Img.open(image_path).convert("RGB")
    except Exception as e:
        _log.error("Failed to open %s: %s", image_path, e)
        return 0, 0

    w, h = img.size
    # Crop bottom stat line — skip the blue bar at the very top of the stat area
    stat_top = int(h * 0.86)
    stat_img = img.crop((0, stat_top, w, h))

    # Erase heart ♥ and sword ⚔ icon pixels before binarization so they
    # don't contaminate digit reads
    from bot.vision import erase_stat_icons
    stat_img = erase_stat_icons(stat_img)
    arr = np.asarray(stat_img, dtype=np.uint8)

    # Dark text → black, rest → white
    gray = np.mean(arr, axis=2)
    bw = np.full(gray.shape, 255, dtype=np.uint8)
    bw[gray < 170] = 0
    col_dark = np.sum(bw == 0, axis=0)

    # Find all gaps (runs of zero dark-pixel columns, >= 3 cols, after x=10)
    gaps = []
    gap_start = None
    for x in range(len(col_dark)):
        if col_dark[x] == 0:
            if gap_start is None:
                gap_start = x
        else:
            if gap_start is not None:
                gap_len = x - gap_start
                if gap_len >= 3 and gap_start > 10:
                    gaps.append((gap_start, x, gap_len))
                gap_start = None

    if not gaps:
        return 0, 0

    # Exclude trailing gap at image edge
    if gaps[-1][1] >= len(col_dark) - 2:
        gaps = gaps[:-1]
    if not gaps:
        return 0, 0

    # Widest gap separates HP section from sword+ATK section
    widest_idx = max(range(len(gaps)), key=lambda i: gaps[i][2])
    hp_end = gaps[widest_idx][0]

    # After the widest gap: find the ATK digits by scanning for a gap→dark transition
    # within the next 55px (sword width + inter-icon gap).  If the sword icon pixels
    # are partially unerased they appear as a dark cluster immediately after widest_end,
    # followed by a small gap, then the ATK digits.  If the sword is fully erased there
    # is no intermediate cluster and widest_end lands directly on the ATK digits.
    widest_end = gaps[widest_idx][1]
    atk_start = widest_end   # fallback: no post-sword gap found
    search_end = min(widest_end + 55, len(col_dark))
    post_gap_start = None
    for _x in range(widest_end, search_end):
        if col_dark[_x] == 0:
            if post_gap_start is None:
                post_gap_start = _x
        else:
            if post_gap_start is not None:
                atk_start = _x   # first dark column after sword + gap = ATK start
                break

    # HP start: skip heart icon — first significant gap separates icon from digits.
    # Minimum 25px to ensure heart icon edge pixels (which can look like '7') are cleared.
    hp_start = gaps[0][1] if gaps[0][2] >= 4 and gaps[0][0] < 55 else 25
    hp_start = max(hp_start, 25)
    if widest_idx == 0:
        hp_start = 25
        hp_end = gaps[0][0]

    hp_region = bw[:, hp_start:hp_end]
    atk_region = bw[:, atk_start:]

    hp = _ocr_stat_region(hp_region)
    atk = _ocr_stat_region(atk_region)
    return hp, atk


def _ocr_stat_region(region: np.ndarray) -> int:
    """OCR a B&W stat region (trimmed, upscaled, padded).

    Uses EasyOCR as the primary engine — it handles this game's pixel-art
    stat font accurately where Tesseract makes systematic errors (5→9, digit
    drop, hallucinated digits).  Tesseract is the fallback for when EasyOCR
    is unavailable or returns zero.
    """
    from PIL import Image as Img

    # Trim to content
    cols = np.sum(region == 0, axis=0)
    nz = np.where(cols > 0)[0]
    if len(nz) == 0:
        return 0
    region = region[:, nz[0]:nz[-1] + 1]

    if region.size == 0 or region.shape[1] < 3:
        return 0

    pil = Img.fromarray(region)
    pil = pil.resize((pil.width * 3, pil.height * 3), Img.NEAREST)
    padded = np.pad(np.asarray(pil), 15, constant_values=255)
    final = Img.fromarray(padded)

    # Primary: EasyOCR — accurate on this game font
    result = 0
    try:
        from bot.easyocr_engine import read_number
        from bot.config import MAX_HP_PER_CAR
        easy_val, easy_conf = read_number(final)
        if easy_val > 0 and easy_conf > 0.3 and easy_val <= MAX_HP_PER_CAR:
            _log.debug("build stat EasyOCR: %d (conf=%.2f)", easy_val, easy_conf)
            result = easy_val
        elif easy_val > MAX_HP_PER_CAR:
            _log.debug("build stat EasyOCR: %d exceeds MAX_HP_PER_CAR — rejecting "
                       "(likely comma/separator misread); falling back to Tesseract", easy_val)
    except Exception as e:
        _log.debug("EasyOCR stat read failed: %s", e)

    # Fallback: Tesseract when EasyOCR unavailable or returned zero
    if result == 0:
        raw_text = _ocr(final, config=_NUMBERS_OCR).strip()
        result = _digits(raw_text)
        _log.debug("build stat Tesseract fallback: raw=%r → %d", raw_text, result)

        # Sanity: reject if digit count exceeds region capacity (~11px/digit).
        # Tesseract sometimes hallucinate extra digits on this font.
        digits_returned = len([c for c in raw_text if c.isdigit()])
        max_digits = max(1, region.shape[1] // 11)
        if result > 0 and digits_returned > max_digits:
            _log.warning("build stat Tesseract: digit count %d > region capacity %d "
                         "(%dpx) — rejecting; raw=%r",
                         digits_returned, max_digits, region.shape[1], raw_text)
            result = 0

    return result


def _ocr_enemy_name(win) -> str:
    """OCR player name from the enemy car detail panel."""
    shot = pyautogui.screenshot(region=win.hud("enemy_name"))
    proc = _prep_name_image(shot)

    raw = _read_name(proc, _NAME_OCR)
    if not raw:
        raw = _read_name(proc, _NAME_OCR_PSM8)

    # EasyOCR fallback for enemy names
    if not raw or raw == "unknown":
        try:
            from bot.easyocr_engine import read_name as _easy_read_name
            easy_text, easy_conf = _easy_read_name(shot, cjk=False)
            if easy_text and easy_conf > 0.3:
                _log.info("enemy_name: EasyOCR: '%s' (conf=%.2f)", easy_text, easy_conf)
                raw = _apply_correction(easy_text)
        except Exception as e:
            _log.debug("enemy_name: EasyOCR unavailable: %s", e)

    return re.sub(r'[\\/:*?"<>|]', '', raw) or "unknown"
