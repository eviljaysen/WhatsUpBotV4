"""tests/test_config.py — Unit tests for bot/config.py (v5.0).

Tests config loading, mtime check, and derived constants.
"""

import os
import json
import tempfile
from unittest.mock import patch

import pytest


class TestConfigMtime:
    def test_reload_skips_when_unchanged(self):
        """reload_config() should skip when mtime hasn't changed."""
        from bot.config import reload_config, CFG

        # First reload always loads
        cfg1 = reload_config()
        assert cfg1 is CFG

        # Second reload with same mtime should skip (fast path)
        cfg2 = reload_config()
        assert cfg2 is CFG

    def test_reload_detects_change(self, tmp_path):
        """reload_config() should reload when mtime changes."""
        import bot.config as config_mod

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"players": {"TEST": "UTC+0"}}))

        orig_path = config_mod._CFG_PATH
        orig_mtime = config_mod._cfg_mtime
        try:
            config_mod._CFG_PATH = str(cfg_file)
            config_mod._cfg_mtime = 0.0  # force reload

            result = config_mod.reload_config()
            assert "TEST" in config_mod.player_timezones
        finally:
            config_mod._CFG_PATH = orig_path
            config_mod._cfg_mtime = orig_mtime
            # Restore original config
            config_mod.reload_config()


class TestDerivedConstants:
    def test_known_names_populated(self):
        from bot.config import _KNOWN_NAMES, player_timezones
        # _KNOWN_NAMES should match player_timezones keys
        assert set(_KNOWN_NAMES) == set(player_timezones.keys())

    def test_upper_map(self):
        from bot.config import _NAME_UPPER_MAP, player_timezones
        for name in player_timezones:
            assert name.upper() in _NAME_UPPER_MAP
            assert _NAME_UPPER_MAP[name.upper()] == name


class TestNamedConstants:
    """Verify named constants exist and have reasonable values."""

    def test_navigation_constants(self):
        from bot.navigation import (
            LOBBY_CENTRE_MARGIN, LOBBY_THROTTLE_SEC,
            LOBBY_HIGH_CONFIDENCE, NEXT_BTN_SEARCH_MARGIN,
            NEXT_BTN_WHITE_THRESH, NEXT_BTN_MAX_ATTEMPTS,
            ADVANCE_MAX_POLLS, ADVANCE_POLL_INTERVAL,
            PIXEL_DIFF_THRESHOLD, SLOT_SETTLE_TIME,
            SCREEN_TRANSITION_TIMEOUT, LOGO_CACHE_MARGIN_PX,
        )
        assert 0 < LOBBY_CENTRE_MARGIN < 0.5
        assert LOBBY_THROTTLE_SEC > 0
        assert 0.5 < LOBBY_HIGH_CONFIDENCE <= 1.0
        assert len(NEXT_BTN_SEARCH_MARGIN) == 2
        assert 200 <= NEXT_BTN_WHITE_THRESH <= 255
        assert NEXT_BTN_MAX_ATTEMPTS > 10
        assert ADVANCE_MAX_POLLS > 10
        assert ADVANCE_POLL_INTERVAL > 0
        assert 1 <= PIXEL_DIFF_THRESHOLD <= 20
        assert SLOT_SETTLE_TIME > 0
        assert SCREEN_TRANSITION_TIMEOUT >= 5
        assert LOGO_CACHE_MARGIN_PX >= 10
