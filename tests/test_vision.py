"""Tests for bot/vision.py — pure image processing, no BlueStacks required."""

import numpy as np
import pytest
from PIL import Image as Img

from bot.vision import convert_to_bw, convert_dark_text, convert_badge_text


def _make_rgb(r, g, b, size=(10, 10)) -> Img.Image:
    """Create a solid-color RGB image."""
    arr = np.full((*size, 3), [r, g, b], dtype=np.uint8)
    return Img.fromarray(arr, 'RGB')


class TestConvertToBw:
    def test_near_white_becomes_black_ink(self):
        img = _make_rgb(240, 245, 255)
        result = np.asarray(convert_to_bw(img))
        assert result.max() == 0   # all pixels become black ink

    def test_non_white_becomes_white_background(self):
        img = _make_rgb(100, 150, 200)
        result = np.asarray(convert_to_bw(img))
        assert result.min() == 255  # all pixels become white background

    def test_paw_blank_value_160_not_triggered(self):
        """Value 160 must NOT trigger the near-white threshold (prevent paw bug)."""
        img = _make_rgb(160, 160, 160)
        result = np.asarray(convert_to_bw(img))
        # 160 < 235, so it should NOT become black ink
        assert result.min() == 255  # white background — correct

    def test_pure_white_255_is_triggered(self):
        """Value 255 DOES trigger convert_to_bw — this is the paw blanking bug."""
        img = _make_rgb(255, 255, 255)
        result = np.asarray(convert_to_bw(img))
        assert result.max() == 0   # becomes black — WHY we use 160 not 255


class TestConvertDarkText:
    def test_dark_pixels_become_black(self):
        img = _make_rgb(100, 100, 100)
        result = np.asarray(convert_dark_text(img))
        assert result.max() == 0

    def test_light_pixels_stay_white(self):
        img = _make_rgb(200, 200, 200)
        result = np.asarray(convert_dark_text(img))
        assert result.min() == 255

    def test_threshold_boundary(self):
        arr = np.full((10, 20, 3), 169, dtype=np.uint8)  # just below 170
        arr[:, 10:, :] = 170  # at threshold
        img = Img.fromarray(arr)
        result = np.asarray(convert_dark_text(img))
        assert result[5, 5] == 0    # 169 < 170 → black
        assert result[5, 15] == 255  # 170 >= 170 → white

    def test_red_text_detected(self):
        """Red score text (dark red) should become black ink."""
        img = _make_rgb(200, 50, 50)  # dark red
        result = np.asarray(convert_dark_text(img))
        assert result.max() == 0  # dark enough to be ink

    def test_blue_text_detected(self):
        """Blue score text should become black ink."""
        img = _make_rgb(50, 80, 180)
        result = np.asarray(convert_dark_text(img))
        assert result.max() == 0

    def test_handles_grayscale_input(self):
        gray = Img.fromarray(np.full((10, 10), 100, dtype=np.uint8))
        result = np.asarray(convert_dark_text(gray))
        assert result.max() == 0


class TestConvertBadgeText:
    def test_white_text_becomes_black_ink(self):
        img = _make_rgb(220, 220, 220)
        result = np.asarray(convert_badge_text(img))
        assert result.max() == 0  # white text → black ink

    def test_colored_background_stays_white(self):
        img = _make_rgb(50, 80, 200)  # blue badge background
        result = np.asarray(convert_badge_text(img))
        assert result.min() == 255  # colored bg → white

    def test_red_badge_background(self):
        img = _make_rgb(200, 50, 50)  # red badge background
        result = np.asarray(convert_badge_text(img))
        assert result.min() == 255
