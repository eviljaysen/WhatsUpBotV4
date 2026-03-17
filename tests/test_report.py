"""Tests for bot/report.py — pure formatting, no BlueStacks required."""

import numpy as np
import pytest
from PIL import Image as Img

from bot.report import parse_timer, _fmt_hm, _fmt_stat, _tz_offset, build_image_grid


class TestParseTimer:
    def test_standard_h_m(self):
        assert parse_timer("17h 23m") == (17, 23)

    def test_uppercase_H(self):
        assert parse_timer("17H 23m") == (17, 23)

    def test_packed_digits(self):
        assert parse_timer("0823") == (8, 23)

    def test_zero_minutes(self):
        assert parse_timer("5h 00m") == (5, 0)

    def test_only_minutes(self):
        h, m = parse_timer("45")
        assert h == 0 and m == 45

    def test_ocr_garbage(self):
        h, m = parse_timer("l7h 2am")   # typical OCR noise
        # Should not raise, and should return something reasonable
        assert isinstance(h, int) and isinstance(m, int)


class TestFmtStat:
    def test_millions(self):
        assert _fmt_stat(2_234_567) == "2.2M"

    def test_thousands(self):
        assert _fmt_stat(639_000) == "639K"

    def test_small(self):
        assert _fmt_stat(42) == "42"

    def test_zero(self):
        assert _fmt_stat(0) == "—"

    def test_negative(self):
        assert _fmt_stat(-1) == "—"

    def test_exactly_1m(self):
        assert _fmt_stat(1_000_000) == "1.0M"


class TestFmtHm:
    def test_basic(self):
        assert _fmt_hm(103) == "01H 43m"

    def test_zero(self):
        assert _fmt_hm(0) == "00H 00m"

    def test_large(self):
        assert _fmt_hm(600) == "10H 00m"


class TestTzOffset:
    def test_positive(self):
        assert _tz_offset("UTC+2") == 2

    def test_negative(self):
        assert _tz_offset("UTC-5") == -5

    def test_gmt(self):
        assert _tz_offset("GMT+1") == 1

    def test_zero(self):
        assert _tz_offset("UTC+0") == 0


class TestBuildImageGrid:
    def test_returns_none_for_empty(self):
        assert build_image_grid([]) is None

    def test_returns_none_for_nonexistent_files(self):
        assert build_image_grid(["/nonexistent/path.png"]) is None

    def test_creates_grid_from_images(self, tmp_path):
        paths = []
        for i in range(4):
            img = Img.fromarray(np.full((100, 150, 3), i * 60, dtype=np.uint8))
            p = tmp_path / f"img_{i}.png"
            img.save(str(p))
            paths.append(str(p))
        grid = build_image_grid(paths, cols=2)
        assert grid is not None
        assert grid.size == (300, 200)  # 2 cols × 150w, 2 rows × 100h
