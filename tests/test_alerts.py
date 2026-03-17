"""tests/test_alerts.py — Unit tests for bot/alerts.py (v5.0)."""

from types import SimpleNamespace

import pytest

from bot.alerts import (
    AlertEvaluator, Alert,
    INSTANT_WARNING, BUILDING_WEAK, SCORE_GAP_CLOSING, CARS_LOST,
)
from bot.analysis import analyze_buildings


# ── Helpers ──────────────────────────────────────────────────────────────────
def _slot(player, building, hp=1000, atk=500, defending=True):
    return SimpleNamespace(
        player=player, building=building,
        hp=hp, atk=atk, defending=defending, screenshot="",
    )


def _meta(team_points=5000, opp_points=3000, team_bonus=50,
          opp_bonus=30, max_points=10000):
    return dict(
        team_points=team_points, opp_points=opp_points,
        team_bonus=team_bonus, opp_bonus=opp_bonus,
        max_points=max_points, opponent="ENEMY",
    )


def _snap(team_pts, opp_pts, timestamp):
    return {"team_pts": team_pts, "opp_pts": opp_pts, "timestamp": timestamp}


def _evaluator(**overrides):
    cfg = {"alert_thresholds": overrides}
    return AlertEvaluator(cfg)


# ═══════════════════════════════════════════════════════════════════════════

class TestInstantWarning:
    def test_triggers_when_close(self):
        ev = _evaluator(instant_warning_minutes=30)
        # opp needs 200 more pts at +30/min = ~6.7 minutes
        meta = _meta(opp_points=9800, opp_bonus=30, max_points=10000)
        analysis = analyze_buildings([])
        alerts = ev.evaluate(meta, analysis)
        instant_alerts = [a for a in alerts if a.alert_type == INSTANT_WARNING]
        assert len(instant_alerts) == 1
        assert "INSTANTS" in instant_alerts[0].message

    def test_no_trigger_when_far(self):
        ev = _evaluator(instant_warning_minutes=30)
        # opp needs 5000 more pts at +30/min = ~167 minutes
        meta = _meta(opp_points=5000, opp_bonus=30, max_points=10000)
        analysis = analyze_buildings([])
        alerts = ev.evaluate(meta, analysis)
        instant_alerts = [a for a in alerts if a.alert_type == INSTANT_WARNING]
        assert len(instant_alerts) == 0

    def test_no_trigger_no_bonus(self):
        ev = _evaluator()
        meta = _meta(opp_bonus=0)
        analysis = analyze_buildings([])
        alerts = ev.evaluate(meta, analysis)
        instant_alerts = [a for a in alerts if a.alert_type == INSTANT_WARNING]
        assert len(instant_alerts) == 0


class TestBuildingWeak:
    def test_triggers_below_threshold(self):
        ev = _evaluator(building_strength_min=2000)
        slots = [_slot("A", 1, hp=500, atk=200)]  # str = 700 < 2000
        analysis = analyze_buildings(slots)
        alerts = ev.evaluate(_meta(), analysis)
        weak = [a for a in alerts if a.alert_type == BUILDING_WEAK]
        assert len(weak) == 1
        assert "Building 1" in weak[0].message

    def test_no_trigger_above_threshold(self):
        ev = _evaluator(building_strength_min=1000)
        slots = [_slot("A", 1, hp=2000, atk=1000)]  # str = 3000 > 1000
        analysis = analyze_buildings(slots)
        alerts = ev.evaluate(_meta(), analysis)
        weak = [a for a in alerts if a.alert_type == BUILDING_WEAK]
        assert len(weak) == 0


class TestScoreGapClosing:
    def test_triggers_on_big_gap(self):
        ev = _evaluator(score_gap_warning=100)
        trajectory = [
            _snap(5000, 3000, 0),
            _snap(5100, 4000, 600),  # opp +1000, us +100 → gap change +900
        ]
        analysis = analyze_buildings([])
        alerts = ev.evaluate(_meta(), analysis, trajectory=trajectory)
        gap = [a for a in alerts if a.alert_type == SCORE_GAP_CLOSING]
        assert len(gap) == 1

    def test_no_trigger_when_ahead(self):
        ev = _evaluator(score_gap_warning=100)
        trajectory = [
            _snap(5000, 3000, 0),
            _snap(6000, 3050, 600),  # us +1000, opp +50
        ]
        analysis = analyze_buildings([])
        alerts = ev.evaluate(_meta(), analysis, trajectory=trajectory)
        gap = [a for a in alerts if a.alert_type == SCORE_GAP_CLOSING]
        assert len(gap) == 0


class TestCarsLost:
    def test_triggers_on_loss(self):
        ev = _evaluator()
        slots = [_slot("A", 1)]
        analysis = analyze_buildings(slots)
        alerts = ev.evaluate(_meta(), analysis, prev_slot_count=5)
        lost = [a for a in alerts if a.alert_type == CARS_LOST]
        assert len(lost) == 1
        assert "4 car(s) lost" in lost[0].message

    def test_no_trigger_when_same(self):
        ev = _evaluator()
        slots = [_slot("A", 1), _slot("B", 2)]
        analysis = analyze_buildings(slots)
        alerts = ev.evaluate(_meta(), analysis, prev_slot_count=2)
        lost = [a for a in alerts if a.alert_type == CARS_LOST]
        assert len(lost) == 0


class TestAlertEvaluatorConfig:
    def test_defaults(self):
        ev = AlertEvaluator({})
        assert ev.instant_warn_min == 30
        assert ev.bldg_str_min == 5_000_000
        assert ev.score_gap_warn == 10_000

    def test_custom_thresholds(self):
        cfg = {
            "alert_thresholds": {
                "instant_warning_minutes": 60,
                "building_strength_min": 1_000_000,
                "score_gap_warning": 5_000,
            }
        }
        ev = AlertEvaluator(cfg)
        assert ev.instant_warn_min == 60
        assert ev.bldg_str_min == 1_000_000
        assert ev.score_gap_warn == 5_000

    def test_webhook_fallback(self):
        ev = AlertEvaluator({"discord_webhook_url": "https://hook.example.com"})
        assert ev.webhook_url == "https://hook.example.com"

    def test_alert_webhook_priority(self):
        ev = AlertEvaluator({
            "alert_webhook_url": "https://alert.example.com",
            "discord_webhook_url": "https://hook.example.com",
        })
        assert ev.webhook_url == "https://alert.example.com"
