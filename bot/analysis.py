"""bot/analysis.py — Strategic analysis engine (v5.0).

Pure computation — no screen capture, no OCR, no pyautogui.
All functions take data structures and return analysis results.
"""

from bot.config import get_logger
from bot.report import _fmt_stat

_log = get_logger("analysis")


# ── Building strength analysis ───────────────────────────────────────────────
def analyze_buildings(slot_results: list) -> dict:
    """Analyze building strength from slot results.

    Args:
        slot_results: list of SlotData instances from a scan

    Returns:
        dict with:
            buildings: {bnum: {total_hp, total_atk, total_str, slot_count,
                               players: [name, ...], defending_count}}
            weakest: int (building number) or None
            strongest: int (building number) or None
            empty: [int] (building numbers with no slots)
    """
    buildings = {}
    for s in slot_results:
        b = buildings.setdefault(s.building, {
            "total_hp": 0, "total_atk": 0, "total_str": 0,
            "slot_count": 0, "players": [], "defending_count": 0,
        })
        b["total_hp"] += s.hp
        b["total_atk"] += s.atk
        b["total_str"] += s.hp + s.atk
        b["slot_count"] += 1
        if s.player not in b["players"]:
            b["players"].append(s.player)
        if s.defending:
            b["defending_count"] += 1

    occupied = [bnum for bnum in range(1, 7) if bnum in buildings]
    empty = [bnum for bnum in range(1, 7) if bnum not in buildings]

    weakest = strongest = None
    if occupied:
        weakest = min(occupied, key=lambda b: buildings[b]["total_str"])
        strongest = max(occupied, key=lambda b: buildings[b]["total_str"])

    return {
        "buildings": buildings,
        "weakest": weakest,
        "strongest": strongest,
        "empty": empty,
    }


def recommend_placements(slot_results: list, player_slots_placed: dict,
                         player_stats: list = None) -> list:
    """Recommend where unplaced players should go.

    Args:
        slot_results: current scan slot results
        player_slots_placed: player → count of slots placed this scan
        player_stats: optional list of dicts with player, total_str keys
                      (from history or build OCR)

    Returns:
        List of (player, building, reason) tuples, strongest players first.
    """
    # Find players with available slots
    unplaced = {}
    for player, placed in player_slots_placed.items():
        avail = 3 - placed
        if avail > 0:
            unplaced[player] = avail

    if not unplaced:
        return []

    # Get building analysis
    analysis = analyze_buildings(slot_results)
    buildings = analysis["buildings"]
    empty = analysis["empty"]

    # Rank unplaced players by strength (strongest first)
    strength_map = {}
    if player_stats:
        strength_map = {s["player"]: s["total_str"] for s in player_stats}

    ranked = sorted(unplaced.keys(),
                    key=lambda p: strength_map.get(p, 0), reverse=True)

    recommendations = []

    # Priority 1: Fill empty buildings
    empty_iter = iter(empty)
    # Priority 2: Reinforce weakest buildings (by total_str)
    weak_order = sorted(
        buildings.keys(),
        key=lambda b: buildings[b]["total_str"]
    ) if buildings else []

    for player in ranked:
        # Try empty building first
        target = next(empty_iter, None)
        if target is not None:
            reason = "empty building — needs defenders"
            recommendations.append((player, target, reason))
            continue

        # Try weakest occupied building
        if weak_order:
            target = weak_order[0]
            bdata = buildings[target]
            reason = (f"weakest building "
                      f"(STR: {_fmt_stat(bdata['total_str'])}, "
                      f"{bdata['slot_count']} slots)")
            recommendations.append((player, target, reason))
            # Don't repeatedly recommend the same building unless it's
            # still the weakest after adding this player
            continue

    return recommendations


# ── Score momentum ───────────────────────────────────────────────────────────
def get_momentum(snapshots: list) -> dict:
    """Calculate score velocity and acceleration from snapshots.

    Args:
        snapshots: list of score_snapshot dicts ordered by timestamp ASC
                   (from get_score_trajectory)

    Returns:
        dict with:
            team_velocity: points per minute (float)
            opp_velocity: points per minute (float)
            team_accel: acceleration (change in velocity, pts/min²)
            opp_accel: acceleration
            team_trend: 'accelerating', 'decelerating', or 'steady'
            opp_trend: same
            snapshots_used: int
    """
    result = {
        "team_velocity": 0.0,
        "opp_velocity": 0.0,
        "team_accel": 0.0,
        "opp_accel": 0.0,
        "team_trend": "steady",
        "opp_trend": "steady",
        "snapshots_used": len(snapshots),
    }

    if len(snapshots) < 2:
        return result

    # Velocity: use first and last snapshot
    first = snapshots[0]
    last = snapshots[-1]
    dt_min = (last["timestamp"] - first["timestamp"]) / 60.0
    if dt_min <= 0:
        return result

    result["team_velocity"] = (last["team_pts"] - first["team_pts"]) / dt_min
    result["opp_velocity"] = (last["opp_pts"] - first["opp_pts"]) / dt_min

    # Acceleration: compare velocity of first half vs second half
    if len(snapshots) >= 3:
        mid = len(snapshots) // 2
        mid_snap = snapshots[mid]

        dt1 = (mid_snap["timestamp"] - first["timestamp"]) / 60.0
        dt2 = (last["timestamp"] - mid_snap["timestamp"]) / 60.0

        if dt1 > 0 and dt2 > 0:
            tv1 = (mid_snap["team_pts"] - first["team_pts"]) / dt1
            tv2 = (last["team_pts"] - mid_snap["team_pts"]) / dt2
            ov1 = (mid_snap["opp_pts"] - first["opp_pts"]) / dt1
            ov2 = (last["opp_pts"] - mid_snap["opp_pts"]) / dt2

            total_dt = (dt1 + dt2)
            result["team_accel"] = (tv2 - tv1) / total_dt if total_dt else 0
            result["opp_accel"] = (ov2 - ov1) / total_dt if total_dt else 0

    # Classify trend
    accel_threshold = 0.5  # pts/min² — below this is "steady"
    for side in ("team", "opp"):
        accel = result[f"{side}_accel"]
        if accel > accel_threshold:
            result[f"{side}_trend"] = "accelerating"
        elif accel < -accel_threshold:
            result[f"{side}_trend"] = "decelerating"

    return result


# ── Formatting ───────────────────────────────────────────────────────────────
def format_building_summary(analysis: dict, alert_thresholds: dict = None) -> str:
    """Format building analysis as a compact table with averages and alerts.

    Args:
        analysis: dict from analyze_buildings()
        alert_thresholds: optional dict with building_strength_min key

    Returns:
        Formatted string table with per-slot averages and alert column.
    """
    buildings = analysis["buildings"]
    if not buildings:
        return ""

    strength_min = 0
    if alert_thresholds:
        strength_min = alert_thresholds.get("building_strength_min", 0)

    lines = []
    lines.append(f"{'BLDG':<5} {'AVG STR':>8} {'AVG HP':>8} {'AVG ATK':>8} {'SLOTS':>5}  ALERT")
    lines.append("-" * 50)

    for bnum in sorted(buildings.keys()):
        b = buildings[bnum]
        n = max(b["slot_count"], 1)
        avg_hp  = b["total_hp"] // n
        avg_atk = b["total_atk"] // n
        avg_str = b["total_str"] // n
        alert = ""
        if bnum == analysis["weakest"]:
            alert = "⚠ WEAK"
        if strength_min and b["total_str"] < strength_min:
            alert = "🔴 LOW"
        lines.append(
            f"B{bnum:<4} "
            f"{_fmt_stat(avg_str):>8} "
            f"{_fmt_stat(avg_hp):>8} "
            f"{_fmt_stat(avg_atk):>8} "
            f"{b['slot_count']:>5}  {alert}"
        )

    if analysis["empty"]:
        for bnum in analysis["empty"]:
            lines.append(f"B{bnum:<4} {'—':>8} {'—':>8} {'—':>8}     0  🔴 EMPTY")

    return "\n".join(lines)



def format_top_players() -> str:
    """Format all players ranked by total strength from build_history/.

    Reads HP/ATK from build_history/ screenshots (stable 3-car record per
    player, updated after each scan). Uses mtime-based OCR cache for speed.
    Compares against SQLite player_stats for change deltas.

    Returns:
        Formatted string table, or empty string if no data.
    """
    from bot.config import BUILD_HISTORY_DIR
    from bot.report import get_player_stats_from_builds

    stats = get_player_stats_from_builds(BUILD_HISTORY_DIR)
    if not stats:
        return ""

    ranked = stats  # all players

    # Load previous stats from DB for delta column (skip garbage values)
    prev_stats = {}
    try:
        from bot.config import MAX_PLAYER_TOTAL
        from bot.history import get_latest_player_stats
        for row in get_latest_player_stats():
            total = row["total_str"]
            if total <= MAX_PLAYER_TOTAL:
                prev_stats[row["player"]] = total
    except Exception as e:
        _log.debug("DB stats not available for delta: %s", e)

    name_w = max(12, max(len(r["player"]) for r in ranked) + 1)
    lines = []
    lines.append(f"{'#':<3} {'PLAYER':<{name_w}} {'TOT STR':>8} {'TOT HP':>8} {'TOT ATK':>8} CARS {'CHG':>6}")
    lines.append("-" * (40 + name_w))

    for i, d in enumerate(ranked, 1):
        name = d["player"]
        # Compute delta vs DB history and classify the change:
        #   ≥5% increase  on full 3-car set → ⚡ possible toolbox buff
        #   >120% increase on full 3-car set → ⚠ likely OCR error (exceeds max buff)
        #   any decrease                    → shown as negative delta only
        delta_str = ""
        flag = ""
        prev = prev_stats.get(name)
        if prev is not None and prev > 0:
            diff = d["total_str"] - prev
            pct = diff / prev          # signed: positive = increase
            if diff > 0:
                delta_str = f"+{_fmt_stat(diff)}"
                if d["cars_found"] == 3:
                    if pct > 1.20:
                        flag = "⚠"    # above max toolbox cap — suspect OCR
                    elif pct >= 0.05:
                        flag = "⚡"   # within toolbox buff range
            elif diff < 0:
                delta_str = f"-{_fmt_stat(abs(diff))}"
        lines.append(
            f"{i:<3} {name:<{name_w}} "
            f"{_fmt_stat(d['total_str']):>8} "
            f"{_fmt_stat(d['total_hp']):>8} "
            f"{_fmt_stat(d['total_atk']):>8} "
            f"{d['cars_found']}    "
            f"{delta_str:>6} {flag}"
        )

    return "\n".join(lines)


def format_momentum(momentum: dict) -> str:
    """Format score momentum as a one-line summary.

    Args:
        momentum: dict from get_momentum()

    Returns:
        Formatted string like "Team +120/min [accelerating] | Opp +95/min [steady]"
    """
    if momentum["snapshots_used"] < 2:
        return ""

    tv = momentum["team_velocity"]
    ov = momentum["opp_velocity"]
    tt = momentum["team_trend"]
    ot = momentum["opp_trend"]

    return (f"Team {tv:+.0f}/min [{tt}] | "
            f"Opp {ov:+.0f}/min [{ot}]")
