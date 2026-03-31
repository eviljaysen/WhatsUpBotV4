"""Tests for bot/ocr.py helpers — no Tesseract or BlueStacks required."""

import pytest
from unittest.mock import patch



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
