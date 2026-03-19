"""bot/alerts.py — Alert evaluation and firing (v5.0).

Evaluates scan results against configurable thresholds and fires alerts
to Discord when conditions are met. No screen capture, no pyautogui.
"""

import time
from dataclasses import dataclass, field

from bot.config import get_logger

_log = get_logger("alerts")


@dataclass
class Alert:
    """A single alert to be posted."""
    alert_type: str   # INSTANT_WARNING, BUILDING_WEAK, SCORE_GAP_CLOSING, CARS_LOST
    severity: str     # 'critical', 'warning', 'info'
    message: str


# ── Alert types ──────────────────────────────────────────────────────────────
INSTANT_WARNING = "INSTANT_WARNING"
BUILDING_WEAK = "BUILDING_WEAK"
SCORE_GAP_CLOSING = "SCORE_GAP_CLOSING"
CARS_LOST = "CARS_LOST"


# ── Default thresholds ──────────────────────────────────────────────────────
_DEFAULTS = {
    "instant_warning_minutes": 30,
    "building_strength_min": 5_000_000,
    "score_gap_warning": 10_000,
}

# Rate-limiting: minimum seconds between alerts of the same type
ALERT_COOLDOWN_SEC = 300  # 5 minutes
_last_fired: dict = {}    # alert_type → timestamp of last fire


class AlertEvaluator:
    """Evaluate scan results against alert thresholds."""

    def __init__(self, cfg: dict):
        thresholds = cfg.get("alert_thresholds", {})
        self.instant_warn_min = thresholds.get(
            "instant_warning_minutes",
            _DEFAULTS["instant_warning_minutes"])
        self.bldg_str_min = thresholds.get(
            "building_strength_min",
            _DEFAULTS["building_strength_min"])
        self.score_gap_warn = thresholds.get(
            "score_gap_warning",
            _DEFAULTS["score_gap_warning"])
        self.webhook_url = cfg.get("alert_webhook_url", "") or cfg.get("discord_webhook_url", "")

    def evaluate(self, meta: dict, analysis: dict,
                 trajectory: list = None,
                 prev_slot_count: int = 0) -> list:
        """Evaluate all alert conditions.

        Args:
            meta: scan metadata (team_points, opp_points, opp_bonus, etc.)
            analysis: dict from analyze_buildings()
            trajectory: list of score snapshots (optional, for momentum)
            prev_slot_count: number of slots in previous scan (for CARS_LOST)

        Returns:
            List of Alert instances.
        """
        alerts = []

        # 1. Instant warning — opponent instants within N minutes
        opp_bonus = meta.get("opp_bonus", 0)
        opp_points = meta.get("opp_points", 0)
        max_points = meta.get("max_points", 0)
        if opp_bonus > 0 and max_points > 0:
            remaining = max_points - opp_points
            mins_to_instant = remaining / opp_bonus
            if 0 < mins_to_instant <= self.instant_warn_min:
                alerts.append(Alert(
                    alert_type=INSTANT_WARNING,
                    severity="critical",
                    message=(f"⚠️ OPPONENT INSTANTS IN ~{mins_to_instant:.0f} MINUTES "
                             f"({opp_points:,} / {max_points:,}, +{opp_bonus}/min)")
                ))

        # 2. Building weak — any building below strength threshold
        buildings = analysis.get("buildings", {})
        for bnum, bdata in buildings.items():
            if 0 < bdata["total_str"] < self.bldg_str_min:
                from bot.report import _fmt_stat
                alerts.append(Alert(
                    alert_type=BUILDING_WEAK,
                    severity="warning",
                    message=(f"🏚️ Building {bnum} is weak: "
                             f"STR {_fmt_stat(bdata['total_str'])} "
                             f"({bdata['slot_count']} slots)")
                ))

        # 3. Score gap closing — opponent gaining on us
        if trajectory and len(trajectory) >= 2:
            latest = trajectory[-1]
            prev = trajectory[-2]
            our_delta = latest["team_pts"] - prev["team_pts"]
            opp_delta = latest["opp_pts"] - prev["opp_pts"]
            gap_change = opp_delta - our_delta
            if gap_change > self.score_gap_warn:
                alerts.append(Alert(
                    alert_type=SCORE_GAP_CLOSING,
                    severity="warning",
                    message=(f"📉 Opponent gaining: they scored +{opp_delta:,} "
                             f"vs our +{our_delta:,} since last scan")
                ))

        # 4. Cars lost — fewer defending slots than previous scan
        current_count = sum(b["slot_count"] for b in buildings.values())
        if prev_slot_count > 0 and current_count < prev_slot_count:
            lost = prev_slot_count - current_count
            alerts.append(Alert(
                alert_type=CARS_LOST,
                severity="warning",
                message=(f"🚗💨 {lost} car(s) lost since last scan! "
                         f"({prev_slot_count} → {current_count})")
            ))

        return alerts

    def fire(self, alerts: list):
        """Post alerts to Discord webhook with per-type rate limiting.

        Each alert type is throttled to at most once per ALERT_COOLDOWN_SEC
        seconds to avoid spamming Discord on repeated scans.
        """
        if not alerts or not self.webhook_url:
            return

        try:
            from bot.discord_post import post_alert
        except ImportError:
            _log.error("discord_post not available")
            return

        now = time.time()
        for alert in alerts:
            last = _last_fired.get(alert.alert_type, 0)
            if now - last < ALERT_COOLDOWN_SEC:
                _log.debug("%s rate-limited (%ds remaining)",
                           alert.alert_type,
                           int(ALERT_COOLDOWN_SEC - (now - last)))
                continue
            _log.info("%s: %s", alert.severity, alert.message)
            try:
                post_alert(alert.message, webhook_url=self.webhook_url)
                _last_fired[alert.alert_type] = now
            except Exception as e:
                _log.error("Failed to post alert: %s", e, exc_info=True)
