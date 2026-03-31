"""tests/test_analysis.py — Unit tests for bot/analysis.py (v5.0).

Pure computation tests — no mocks needed.
"""

from types import SimpleNamespace

import pytest

from bot.analysis import (
    analyze_buildings,
    recommend_placements,
    get_momentum,
    format_building_summary,
    format_top_players,
    format_momentum,
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _slot(player, building, hp=1000, atk=500, defending=True):
    return SimpleNamespace(
        player=player, building=building,
        hp=hp, atk=atk, defending=defending, screenshot="",
    )


def _snap(team_pts, opp_pts, timestamp):
    return {
        "team_pts": team_pts, "opp_pts": opp_pts,
        "timestamp": timestamp,
        "team_bonus": 0, "opp_bonus": 0,
        "timer_h": 0, "timer_m": 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Building analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyzeBuildings:
    def test_basic_totals(self):
        slots = [
            _slot("ALICE", 1, hp=1000, atk=500),
            _slot("BOB", 1, hp=2000, atk=800),
            _slot("CAROL", 2, hp=3000, atk=1000),
        ]
        result = analyze_buildings(slots)
        b1 = result["buildings"][1]
        assert b1["total_hp"] == 3000
        assert b1["total_atk"] == 1300
        assert b1["total_str"] == 4300
        assert b1["slot_count"] == 2
        assert set(b1["players"]) == {"ALICE", "BOB"}

    def test_weakest_strongest(self):
        slots = [
            _slot("A", 1, hp=100, atk=50),
            _slot("B", 2, hp=5000, atk=3000),
        ]
        result = analyze_buildings(slots)
        assert result["weakest"] == 1
        assert result["strongest"] == 2

    def test_empty_buildings(self):
        slots = [_slot("A", 3, hp=1000, atk=500)]
        result = analyze_buildings(slots)
        assert set(result["empty"]) == {1, 2, 4, 5, 6}

    def test_no_slots(self):
        result = analyze_buildings([])
        assert result["weakest"] is None
        assert result["strongest"] is None
        assert len(result["empty"]) == 6

    def test_defending_count(self):
        slots = [
            _slot("A", 1, defending=True),
            _slot("B", 1, defending=False),
            _slot("C", 1, defending=True),
        ]
        result = analyze_buildings(slots)
        assert result["buildings"][1]["defending_count"] == 2

    def test_duplicate_player_not_repeated(self):
        slots = [
            _slot("ALICE", 1, hp=1000),
            _slot("ALICE", 1, hp=2000),  # same player, same building
        ]
        result = analyze_buildings(slots)
        assert result["buildings"][1]["players"] == ["ALICE"]
        assert result["buildings"][1]["slot_count"] == 2


class TestRecommendPlacements:
    def test_recommend_to_empty_building(self):
        slots = [_slot("ALICE", 1)]
        placed = {"ALICE": 3, "BOB": 0}  # ALICE fully placed
        recs = recommend_placements(slots, placed)
        assert len(recs) == 1
        assert recs[0][0] == "BOB"
        assert recs[0][1] in [2, 3, 4, 5, 6]  # any empty building
        assert "empty" in recs[0][2]

    def test_no_unplaced(self):
        slots = [_slot("A", 1)]
        placed = {"A": 3}
        recs = recommend_placements(slots, placed)
        assert recs == []

    def test_strongest_player_first(self):
        slots = [_slot("X", 1)]
        placed = {"WEAK": 0, "STRONG": 0, "X": 3}
        stats = [
            {"player": "STRONG", "total_str": 10000},
            {"player": "WEAK", "total_str": 1000},
        ]
        recs = recommend_placements(slots, placed, player_stats=stats)
        assert len(recs) == 2
        assert recs[0][0] == "STRONG"
        assert recs[1][0] == "WEAK"

    def test_recommend_to_weakest_when_no_empty(self):
        slots = [
            _slot("A", 1, hp=100), _slot("A", 2, hp=100),
            _slot("A", 3, hp=100), _slot("A", 4, hp=100),
            _slot("A", 5, hp=100), _slot("A", 6, hp=5000),
        ]
        placed = {"A": 3, "B": 0}
        recs = recommend_placements(slots, placed)
        assert len(recs) == 1
        # Should recommend one of the weak buildings (1-5), not B6
        assert recs[0][1] != 6


# ═══════════════════════════════════════════════════════════════════════════
# Score momentum
# ═══════════════════════════════════════════════════════════════════════════

class TestGetMomentum:
    def test_two_snapshots(self):
        snaps = [
            _snap(1000, 500, 0),
            _snap(2000, 800, 600),  # 10 minutes later
        ]
        m = get_momentum(snaps)
        assert abs(m["team_velocity"] - 100.0) < 0.01  # 1000 pts / 10 min
        assert abs(m["opp_velocity"] - 30.0) < 0.01
        assert m["team_trend"] == "steady"  # only 2 snapshots, no accel

    def test_acceleration(self):
        # Team accelerating: slow then fast
        snaps = [
            _snap(0, 0, 0),
            _snap(100, 100, 600),    # +100 in 10min = 10/min
            _snap(1100, 200, 1200),  # +1000 in 10min = 100/min
        ]
        m = get_momentum(snaps)
        assert m["team_trend"] == "accelerating"

    def test_deceleration(self):
        # Team decelerating: fast then slow
        snaps = [
            _snap(0, 0, 0),
            _snap(1000, 100, 600),   # +1000 in 10min = 100/min
            _snap(1100, 200, 1200),  # +100 in 10min = 10/min
        ]
        m = get_momentum(snaps)
        assert m["team_trend"] == "decelerating"

    def test_single_snapshot(self):
        m = get_momentum([_snap(1000, 500, 0)])
        assert m["team_velocity"] == 0.0
        assert m["snapshots_used"] == 1

    def test_empty(self):
        m = get_momentum([])
        assert m["snapshots_used"] == 0

    def test_same_timestamp(self):
        snaps = [_snap(0, 0, 100), _snap(1000, 500, 100)]
        m = get_momentum(snaps)
        assert m["team_velocity"] == 0.0  # dt=0, no division error


# ═══════════════════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatBuildingSummary:
    def test_basic_format_shows_averages(self):
        slots = [
            _slot("A", 1, hp=1_200_000, atk=800_000),
            _slot("B", 1, hp=800_000, atk=200_000),
            _slot("C", 2, hp=500_000, atk=300_000),
        ]
        analysis = analyze_buildings(slots)
        text = format_building_summary(analysis)
        assert "B1" in text
        assert "B2" in text
        assert "AVG" in text
        # B1: 2 slots, avg_hp = 1M, avg_atk = 500K
        assert "1.0M" in text   # avg HP for B1
        assert "⚠ WEAK" in text  # weakest marker on B2

    def test_empty_returns_empty(self):
        analysis = analyze_buildings([])
        assert format_building_summary(analysis) == ""

    def test_alert_column_with_thresholds(self):
        slots = [_slot("A", 1, hp=100, atk=50)]
        analysis = analyze_buildings(slots)
        text = format_building_summary(analysis,
                                       alert_thresholds={"building_strength_min": 1000})
        assert "🔴 LOW" in text

    def test_empty_buildings_show_empty_alert(self):
        slots = [_slot("A", 1)]
        analysis = analyze_buildings(slots)
        text = format_building_summary(analysis)
        assert "🔴 EMPTY" in text


class TestFormatTopPlayers:
    """format_top_players reads from build_history/ — these tests mock the
    get_player_stats_from_builds return value to avoid needing real images."""

    def test_basic_ranking(self):
        mock_stats = [
            {"player": "ALICE", "total_hp": 5_000_000, "total_atk": 1_500_000,
             "total_str": 6_500_000, "cars_found": 2,
             "hp_1": 3_000_000, "atk_1": 1_000_000,
             "hp_2": 2_000_000, "atk_2": 500_000,
             "hp_3": 0, "atk_3": 0},
            {"player": "BOB", "total_hp": 1_000_000, "total_atk": 200_000,
             "total_str": 1_200_000, "cars_found": 1,
             "hp_1": 1_000_000, "atk_1": 200_000,
             "hp_2": 0, "atk_2": 0, "hp_3": 0, "atk_3": 0},
        ]
        from unittest.mock import patch
        with patch("bot.report.get_player_stats_from_builds", return_value=mock_stats):
            text = format_top_players()
        assert "ALICE" in text
        assert "BOB" in text
        data_lines = [l for l in text.strip().split("\n")
                      if l.strip() and not l.startswith("-")
                      and not l.startswith("#") and "PLAYER" not in l]
        assert "ALICE" in data_lines[0]

    def test_empty(self):
        from unittest.mock import patch
        with patch("bot.report.get_player_stats_from_builds", return_value=[]):
            assert format_top_players() == ""

    def test_all_players_shown(self):
        """format_top_players shows all players (no limit)."""
        mock_stats = [
            {"player": f"P{i}", "total_hp": 1000 * i, "total_atk": 500,
             "total_str": 1000 * i + 500, "cars_found": 1,
             "hp_1": 1000 * i, "atk_1": 500,
             "hp_2": 0, "atk_2": 0, "hp_3": 0, "atk_3": 0}
            for i in range(14, 0, -1)
        ]
        from unittest.mock import patch
        with patch("bot.report.get_player_stats_from_builds", return_value=mock_stats):
            text = format_top_players()
        data_lines = [l for l in text.split("\n")
                      if l.strip() and not l.startswith("-")
                      and not l.startswith("#") and "PLAYER" not in l]
        assert len(data_lines) == 14   # all 14 players shown


class TestFormatMomentum:
    def test_basic(self):
        m = {
            "team_velocity": 120.5,
            "opp_velocity": -30.0,
            "team_trend": "accelerating",
            "opp_trend": "decelerating",
            "snapshots_used": 5,
        }
        text = format_momentum(m)
        assert "+121/min" in text or "+120/min" in text
        assert "accelerating" in text
        assert "decelerating" in text

    def test_insufficient_data(self):
        m = {"snapshots_used": 1}
        assert format_momentum(m) == ""
