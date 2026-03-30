"""bot/vision.py — Pure image processing.

No OCR calls, no screen capture, no pyautogui. All functions take PIL Images
or numpy arrays and return PIL Images or numpy arrays. Fully testable offline.
"""

import numpy as np
from PIL import Image as Img


def convert_to_bw(image, white_threshold=(235, 242, 254)) -> Img.Image:
    """Near-white pixels → black ink; everything else → white background.

    Tuned for the game's HUD font: near-white (R≥235, G≥242, B≥254) becomes
    black ink so Tesseract sees dark text on white.

    IMPORTANT: The paw-blanking value must be 160 (mid-gray), NOT 255.
    Pure white (255) triggers this threshold and becomes black ink, producing
    spurious OCR characters from the blanked area.
    """
    rgb  = np.asarray(image, dtype=np.uint8)
    mask = ((rgb[:, :, 0] >= white_threshold[0]) &
            (rgb[:, :, 1] >= white_threshold[1]) &
            (rgb[:, :, 2] >= white_threshold[2]))
    out       = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    out[mask] = 0
    return Img.fromarray(out)


def convert_white_text(image) -> Img.Image:
    """Generic white text (≥200 all channels) → black ink.

    Less strict than convert_to_bw; used for max_points region which may
    have slightly different white values.
    """
    rgb  = np.asarray(image, dtype=np.uint8)
    mask = (rgb[:, :, 0] >= 200) & (rgb[:, :, 1] >= 200) & (rgb[:, :, 2] >= 200)
    out  = np.full((rgb.shape[0], rgb.shape[1]), 255, dtype=np.uint8)
    out[mask] = 0
    return Img.fromarray(out)


def convert_dark_text(image, threshold=170) -> Img.Image:
    """Dark pixels → black ink, light pixels → white background.

    A universal converter for any dark-on-light text. Works regardless
    of text color (red, blue, green, brown) — only cares about luminance.
    More robust than color-specific converters like convert_numbers.

    Args:
        image: PIL Image (RGB or grayscale)
        threshold: grayscale cutoff — pixels below this become black ink
    """
    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    out = np.full(arr.shape, 255, dtype=np.uint8)
    out[arr < threshold] = 0
    return Img.fromarray(out)


def erase_stat_icons(image) -> Img.Image:
    """Blank out heart ♥ and sword ⚔ icon pixels before stat OCR.

    Replaces colored icon pixels with white so they don't contaminate digit
    reads. Targets the specific colors used by the game:
      - Heart: red  (R≥180, G<100, B<100)
      - Sword: blue/teal (B≥140, R<130, G≥80)

    Returns a new RGB PIL Image with icon pixels set to white (255,255,255).
    Safe to call on any image — non-matching pixels are unchanged.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    heart_mask = (R >= 180) & (G < 100) & (B < 100)
    sword_mask = (B >= 140) & (R < 130) & (G >= 80)
    icon_mask  = heart_mask | sword_mask

    arr[icon_mask] = 255
    return Img.fromarray(arr)


def convert_badge_text(image) -> Img.Image:
    """Extract white +NNN text from a colored bonus badge.

    The badge has bright white text on a blue or red background.
    Simply extracts white pixels (all channels >= 180) as black ink.
    Simpler and more robust than blob-based _extract_badge_text.
    """
    rgb = np.asarray(image, dtype=np.uint8)
    mask = (rgb[:, :, 0] >= 180) & (rgb[:, :, 1] >= 180) & (rgb[:, :, 2] >= 180)
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    out[mask] = 0
    return Img.fromarray(out)


