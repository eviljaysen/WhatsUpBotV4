"""bot/easyocr_engine.py — Lazy-loaded EasyOCR wrapper.

EasyOCR is a heavy dependency (~2GB with PyTorch). This module ensures it is
only imported and initialized on first use. Two reader instances are cached:
one for ASCII-only ('en') and one for CJK ('ja' + 'en').

All public functions degrade gracefully when easyocr is not installed —
they return empty results so callers never need to check availability first.
"""

import numpy as np
from PIL import Image as Img

from bot.config import CFG

# Cached reader instances — created on first use
_reader_ascii = None
_reader_cjk = None
_available = None  # None = not checked yet


def is_available() -> bool:
    """Check if easyocr can be imported (without loading models)."""
    global _available
    if _available is None:
        try:
            import easyocr  # noqa: F401
            _available = True
        except ImportError:
            _available = False
            print("[easyocr] Not installed — fallback OCR disabled")
    return _available


def _get_reader(cjk: bool = False):
    """Create or return a cached EasyOCR Reader.

    Args:
        cjk: If True, load Japanese + English. Otherwise English only.

    Returns:
        easyocr.Reader instance, or None if not available.
    """
    global _reader_ascii, _reader_cjk

    if not is_available():
        return None
    if not CFG.get("easyocr_enabled", True):
        return None

    import warnings
    import easyocr
    warnings.filterwarnings("ignore", message=".*pin_memory.*")

    gpu = CFG.get("easyocr_gpu", False)

    if cjk:
        if _reader_cjk is None:
            print("[easyocr] Loading CJK reader (ja+en)… this may take a moment")
            _reader_cjk = easyocr.Reader(['ja', 'en'], gpu=gpu, verbose=False)
            print("[easyocr] CJK reader ready")
        return _reader_cjk
    else:
        if _reader_ascii is None:
            print("[easyocr] Loading ASCII reader (en)… this may take a moment")
            _reader_ascii = easyocr.Reader(['en'], gpu=gpu, verbose=False)
            print("[easyocr] ASCII reader ready")
        return _reader_ascii


def read_text(image, cjk: bool = False, allowlist: str = None,
              paragraph: bool = False) -> list:
    """Run EasyOCR on a PIL Image or numpy array.

    Args:
        image: PIL Image or numpy array (RGB or grayscale)
        cjk: Use CJK reader (Japanese + English)
        allowlist: Restrict character set (e.g. '0123456789')
        paragraph: Merge results into paragraph mode

    Returns:
        List of (bbox, text, confidence) tuples, sorted by confidence desc.
        Empty list if EasyOCR is not installed or fails.
    """
    reader = _get_reader(cjk=cjk)
    if reader is None:
        return []

    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Img.Image):
            arr = np.asarray(image)
        else:
            arr = image

        results = reader.readtext(
            arr,
            allowlist=allowlist,
            paragraph=paragraph,
        )
        # Sort by confidence descending
        results.sort(key=lambda r: r[2], reverse=True)
        return results
    except Exception as e:
        print(f"[easyocr] OCR error: {e}")
        return []


def read_number(image, cjk: bool = False) -> tuple:
    """OCR digits from image, return (int_value, confidence).

    Uses allowlist='0123456789+' to constrain output.

    Returns:
        (value: int, confidence: float). (0, 0.0) on failure.
    """
    import re

    results = read_text(image, cjk=cjk, allowlist='0123456789+')
    if not results:
        return 0, 0.0

    # Combine all detected text fragments
    all_text = ' '.join(r[1] for r in results)
    digits = re.sub(r'[^0-9]', '', all_text)
    if not digits:
        return 0, 0.0

    # Use highest confidence from any result
    best_conf = max(r[2] for r in results)
    return int(digits), best_conf


def read_name(image, cjk: bool = False) -> tuple:
    """OCR a player name from image, return (text, confidence).

    Returns:
        (name: str, confidence: float). ('', 0.0) on failure.
    """
    results = read_text(image, cjk=cjk)
    if not results:
        return '', 0.0

    # Take the highest-confidence result
    best = results[0]
    return best[1].strip(), best[2]
