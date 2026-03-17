"""Tests for bot/ocr.py helpers — no Tesseract or BlueStacks required."""

import pytest
from unittest.mock import patch


class TestResolvesCjk:
    """Tests for _resolve_cjk — CJK garbage prefix stripping + fuzzy match."""

    def _run(self, raw_cjk, corrections):
        """Run _resolve_cjk with a mock _OCR_CORRECTIONS dict."""
        with patch("bot.ocr._OCR_CORRECTIONS", corrections), \
             patch("bot.ocr.CFG", {"cjk_match_cutoff": 0.65}):
            from bot.ocr import _resolve_cjk
            return _resolve_cjk(raw_cjk)

    def test_exact_match(self):
        result = self._run("鮭とば", {"鮭とば": "SAKETOBA"})
        assert result == "SAKETOBA"

    def test_garbage_prefix_stripped(self):
        # 'し りぷとん' — first word is garbage, second is the real name
        result = self._run("し りぷとん", {"りぷとん": "LIPTON"})
        assert result == "LIPTON"

    def test_fuzzy_match_single_char_variation(self):
        # ょ → よ style OCR variation — fuzzy should catch at 0.65
        result = self._run("ぴょきち", {"ぴよきち": "PIYO"})
        assert result == "PIYO"

    def test_empty_string(self):
        result = self._run("", {"鮭とば": "SAKETOBA"})
        assert result == ""

    def test_no_match_returns_cleaned(self):
        result = self._run("unknown text", {"鮭とば": "SAKETOBA"})
        assert result == "unknown text"


class TestDigits:
    def test_pure_digits(self):
        from bot.ocr import _digits
        assert _digits("12345") == 12345

    def test_mixed(self):
        from bot.ocr import _digits
        assert _digits("12,345 pts") == 12345

    def test_empty(self):
        from bot.ocr import _digits
        assert _digits("") == 0

    def test_no_digits(self):
        from bot.ocr import _digits
        assert _digits("abc") == 0
