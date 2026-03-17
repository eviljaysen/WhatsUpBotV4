"""tests/test_history.py — Unit tests for bot/history.py (v5.0).

Uses an in-memory SQLite database so tests are fast and isolated.
"""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Patch DB_PATH before importing history so _conn() uses :memory:
_TEST_DB_PATH = ":memory:"

# We need a persistent connection for :memory: — otherwise each _conn() call
# creates a fresh (empty) database.  Override _conn to reuse one connection.
import sqlite3
import contextlib

_shared_con = None


@contextlib.contextmanager
def _test_conn():
    """Reuse a single in-memory connection across all calls within a test."""
    global _shared_con
    if _shared_con is None:
        _shared_con = sqlite3.connect(":memory:", check_same_thread=False)
        _shared_con.row_factory = sqlite3.Row
    try:
        yield _shared_con
        _shared_con.commit()
    except Exception:
        _shared_con.rollback()
        raise


@pytest.fixture(autouse=True)
def _fresh_db():
    """Give each test a fresh in-memory database."""
    global _shared_con
    _shared_con = None
    with patch("bot.history._conn", _test_conn), \
         patch("bot.history.DB_PATH", ":memory:"):
        yield


# ── Now safe to import ──────────────────────────────────────────────────────
from bot.history import (
    init_db, save_scan, save_player_stats,
    get_latest_player_stats, get_player_stats_history,
    get_last_scan, get_player_hp_trend, get_opponent_history,
    get_or_create_war, save_score_snapshot, get_score_trajectory,
    close_war, save_enemy_slot, get_opponent_roster,
    get_war_history, get_war_by_id,
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _slot(player, building, hp=1000, atk=500, defending=True, screenshot=""):
    return SimpleNamespace(
        player=player, building=building,
        hp=hp, atk=atk, defending=defending, screenshot=screenshot,
    )


def _meta(opponent="ENEMY", team_points=5000, opp_points=3000,
          team_bonus=50, opp_bonus=30, max_points=10000,
          timer_h=12, timer_m=30):
    return dict(
        opponent=opponent, team_points=team_points, opp_points=opp_points,
        team_bonus=team_bonus, opp_bonus=opp_bonus, max_points=max_points,
        timer_h=timer_h, timer_m=timer_m,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Existing v4.0 functionality
# ═══════════════════════════════════════════════════════════════════════════

class TestInitDb:
    def test_creates_tables(self):
        init_db()
        # Should not raise on second call
        init_db()

    def test_tables_exist(self):
        init_db()
        global _shared_con
        tables = {r[0] for r in _shared_con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "scans" in tables
        assert "slots" in tables
        assert "player_stats" in tables
        assert "wars" in tables
        assert "score_snapshots" in tables
        assert "opponent_players" in tables


class TestSaveScan:
    def test_save_and_retrieve(self):
        init_db()
        meta = _meta()
        slots = [_slot("ALICE", 1), _slot("BOB", 2)]
        scan_id = save_scan(meta, slots)
        assert scan_id >= 1

        prev_meta, prev_slots = get_last_scan("ENEMY")
        assert prev_meta is not None
        assert prev_meta["team_pts"] == 5000
        assert prev_meta["opp_pts"] == 3000
        assert len(prev_slots) == 2

    def test_no_scan_returns_none(self):
        init_db()
        meta, slots = get_last_scan("NOBODY")
        assert meta is None
        assert slots == []


class TestPlayerStats:
    def test_save_and_query(self):
        init_db()
        meta = _meta()
        slots = [_slot("ALICE", 1), _slot("ALICE", 2)]
        scan_id = save_scan(meta, slots)

        three_slots = [
            _slot("ALICE", 1, hp=1000, atk=500),
            _slot("ALICE", 2, hp=2000, atk=600),
            _slot("ALICE", 3, hp=3000, atk=700),
        ]
        save_player_stats(scan_id, int(time.time()), "ALICE", three_slots)

        latest = get_latest_player_stats()
        assert len(latest) == 1
        assert latest[0]["player"] == "ALICE"
        assert latest[0]["total_hp"] == 6000
        assert latest[0]["total_atk"] == 1800
        assert latest[0]["total_str"] == 7800

    def test_hp_trend(self):
        init_db()
        meta = _meta()
        scan_id = save_scan(meta, [])

        base_ts = 1000000
        for i, hp_val in enumerate([1000, 2000, 3000]):
            slots = [
                _slot("BOB", 1, hp=hp_val, atk=100),
                _slot("BOB", 2, hp=hp_val, atk=100),
                _slot("BOB", 3, hp=hp_val, atk=100),
            ]
            save_player_stats(scan_id, base_ts + i, "BOB", slots)

        trend = get_player_hp_trend("BOB", limit=3)
        assert len(trend) == 3
        # Most recent first
        assert trend[0] == 9000
        assert trend[2] == 3000

    def test_stats_history(self):
        init_db()
        scan_id = save_scan(_meta(), [])
        for i in range(3):
            slots = [
                _slot("CAROL", 1, hp=1000 * (i + 1)),
                _slot("CAROL", 2, hp=1000 * (i + 1)),
                _slot("CAROL", 3, hp=1000 * (i + 1)),
            ]
            save_player_stats(scan_id, int(time.time()) + i, "CAROL", slots)

        history = get_player_stats_history("CAROL", limit=2)
        assert len(history) == 2


class TestOpponentHistory:
    def test_returns_scans(self):
        init_db()
        save_scan(_meta("FOE_A"), [])
        save_scan(_meta("FOE_A"), [])
        save_scan(_meta("FOE_B"), [])

        hist = get_opponent_history("FOE_A")
        assert len(hist) == 2


# ═══════════════════════════════════════════════════════════════════════════
# New v5.0 war tracking
# ═══════════════════════════════════════════════════════════════════════════

class TestWarTracking:
    def test_create_war(self):
        init_db()
        war_id = get_or_create_war("WOLVES")
        assert war_id >= 1

    def test_reuse_ongoing_war(self):
        init_db()
        w1 = get_or_create_war("WOLVES")
        w2 = get_or_create_war("WOLVES")
        assert w1 == w2

    def test_new_war_after_close(self):
        init_db()
        w1 = get_or_create_war("WOLVES")
        close_war(w1, "win", 50000, 30000)
        w2 = get_or_create_war("WOLVES")
        assert w2 != w1

    def test_close_war(self):
        init_db()
        war_id = get_or_create_war("BEARS")
        close_war(war_id, "loss", 20000, 40000)
        war = get_war_by_id(war_id)
        assert war["result"] == "loss"
        assert war["our_final"] == 20000
        assert war["opp_final"] == 40000
        assert war["end_ts"] is not None

    def test_war_history(self):
        init_db()
        w1 = get_or_create_war("ALPHA")
        close_war(w1, "win")
        w2 = get_or_create_war("BETA")
        close_war(w2, "loss")

        history = get_war_history(limit=10)
        assert len(history) == 2


class TestScoreSnapshots:
    def test_save_and_query(self):
        init_db()
        war_id = get_or_create_war("FOXES")
        scan_id = save_scan(_meta("FOXES"), [])
        save_score_snapshot(scan_id, war_id, _meta(
            team_points=5000, opp_points=3000, timer_h=10, timer_m=15))

        traj = get_score_trajectory(war_id)
        assert len(traj) == 1
        assert traj[0]["team_pts"] == 5000
        assert traj[0]["opp_pts"] == 3000

    def test_multiple_snapshots_ordered(self):
        init_db()
        war_id = get_or_create_war("FOXES")

        for pts in [1000, 3000, 5000]:
            scan_id = save_scan(_meta("FOXES", team_points=pts), [])
            save_score_snapshot(scan_id, war_id, _meta(team_points=pts))
            time.sleep(0.01)

        traj = get_score_trajectory(war_id)
        assert len(traj) == 3
        # Ascending order
        assert traj[0]["team_pts"] == 1000
        assert traj[2]["team_pts"] == 5000


class TestOpponentPlayers:
    def test_save_and_roster(self):
        init_db()
        war_id = get_or_create_war("HAWKS")
        save_enemy_slot(war_id, "HAWKS", "PLAYER_A", 1, hp=5000, atk=2000)
        save_enemy_slot(war_id, "HAWKS", "PLAYER_B", 2, hp=3000, atk=1000)

        roster = get_opponent_roster("HAWKS")
        assert len(roster) == 2
        # Sorted by strength desc
        assert roster[0]["player_name"] == "PLAYER_A"
        assert roster[1]["player_name"] == "PLAYER_B"

    def test_roster_latest_per_player(self):
        init_db()
        war_id = get_or_create_war("HAWKS")
        save_enemy_slot(war_id, "HAWKS", "PLAYER_A", 1, hp=1000, timestamp=1000)
        save_enemy_slot(war_id, "HAWKS", "PLAYER_A", 2, hp=9000, timestamp=2000)

        roster = get_opponent_roster("HAWKS")
        assert len(roster) == 1
        assert roster[0]["hp"] == 9000  # most recent

    def test_empty_roster(self):
        init_db()
        roster = get_opponent_roster("NOBODY")
        assert roster == []
