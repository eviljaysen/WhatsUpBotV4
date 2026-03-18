"""bot/report.py — Report building, formatting, and player stats extraction.

Builds formatted reports from scan data and extracts player stats from
saved build images via OCR. No screen capture, no pyautogui.
"""

import math
import os
import re
import time
from datetime import datetime, timedelta

import pytz

from bot.config import (
    player_timezones, BOT_SLOTS_TOTAL, SLEEP_START, SLEEP_END
)


# ── Formatting helpers ─────────────────────────────────────────────────────────
def _fmt_hm(minutes: int) -> str:
    """Format integer minutes as 'HHH HMm'. E.g. 103 → '01H 43m'."""
    h, m = divmod(minutes, 60)
    return f"{h:02d}H {m:02d}m"


def _fmt_stat(n: int) -> str:
    """Format a large integer as compact notation for table columns.

    Examples: 2_234_567 → '2.2M', 639_000 → '639K', 0 → '—'
    """
    if n <= 0:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def parse_timer(text: str) -> tuple:
    """Return (hours, minutes) from a timer string.

    Handles: '17h 23m', '0823', '8 23', '17H23m', etc.
    Falls back gracefully on OCR garbage.
    """
    text = re.sub(r'[^0-9h ]', '', text.lower())
    if 'h' in text:
        parts = re.split(r'h\s*', text)
        h = _digits(parts[0])
        m = _digits(parts[1]) if len(parts) > 1 else 0
        return min(h, 48), min(m, 59)
    total = _digits(text)
    if total > 99:
        h_guess = total // 100
        m_guess = total % 100
        if h_guess <= 48 and m_guess <= 59:
            return h_guess, m_guess
    return 0, min(total, 59)


def _digits(text: str) -> int:
    import re as _re
    d = _re.sub(r'[^0-9]', '', text)
    return int(d) if d else 0


def _tz_offset(tz_str: str) -> int:
    """Return signed integer hours from a 'GMT±N' or 'UTC±N' string."""
    try:
        sign = -1 if '-' in tz_str else 1
        return sign * int(re.sub(r'[^0-9]', '', tz_str))
    except Exception:
        return 0


# ── Main report builder ────────────────────────────────────────────────────────
def build_report(ctx, scores: dict, h: int, m: int, avail_only: bool = False) -> tuple:
    """Build the full Discord-formatted report string.

    Args:
        ctx: ScanContext (uses slot_results, players_dict)
        scores: dict with team_points, opp_points, team_bonus, opp_bonus,
                max_points, opponent
        h, m: timer hours and minutes

    Returns:
        (report_str, meta_dict)
    """
    team_points = scores["team_points"]
    opp_points  = scores["opp_points"]
    team_bonus  = scores["team_bonus"]
    opp_bonus   = scores["opp_bonus"]
    max_points  = scores["max_points"]
    opponent    = scores["opponent"]

    timer_left = h * 60 + m
    now_ts     = int(time.time())

    team_time_m = opp_time_m = 0
    team_instant = opp_instant = 0
    if team_bonus > 0:
        team_time_m  = math.ceil((max_points - team_points) / team_bonus)
        team_instant = int(time.time() + team_time_m * 60)
    if opp_bonus > 0:
        opp_time_m  = math.ceil((max_points - opp_points) / opp_bonus)
        opp_instant = int(time.time() + opp_time_m * 60)

    proj_mins = timer_left
    if team_time_m > 0: proj_mins = min(proj_mins, team_time_m)
    if opp_time_m  > 0: proj_mins = min(proj_mins, opp_time_m)

    team_icon = "🟢" if team_points >= opp_points else "🔴"
    opp_icon  = "🔴" if team_points >= opp_points else "🟢"

    # ── Compute awake/asleep for each player ─────────────────────────────────
    utc_now = datetime.now(pytz.utc)

    def _player_status(name):
        tz_str = player_timezones.get(name)
        if not tz_str:
            return False, ""
        local_dt = utc_now + timedelta(hours=_tz_offset(tz_str))
        asleep   = local_dt.hour >= SLEEP_START or local_dt.hour < SLEEP_END
        return asleep, local_dt.strftime("%H:%M")

    # ── Player stats table ────────────────────────────────────────────────────
    from collections import defaultdict
    player_slots: dict = defaultdict(list)
    for s in ctx.slot_results:
        player_slots[s.player].append(s)

    table_data = []
    for player in player_timezones:
        slots = player_slots.get(player, [])
        if slots:
            buildings = sorted(s.building for s in slots)
            bldg_str  = ",".join(str(b) for b in buildings)
            defending = all(s.defending for s in slots)
        else:
            bldg_str  = "—"
            defending = True
        asleep, loc_time = _player_status(player)
        placed = ctx.players_dict.get(player, 0)
        avail  = 3 - placed
        table_data.append((player, bldg_str, defending, asleep, loc_time, avail))

    if avail_only:
        table_data = [r for r in table_data if r[5] > 0]  # r[5] = avail

    table_data.sort(key=lambda r: (r[3], r[0]))

    table = ""
    if table_data:
        name_w  = max(12, max(len(r[0]) for r in table_data) + 2)
        bldg_w  = max(5, max(len(r[1]) for r in table_data) + 1)
        # Emoji = 2 visual chars but 1 in len(); pad header 1 extra to compensate
        header  = (f"  TIME       "
                   f"{'PLAYER':<{name_w}} {'BLDGS':<{bldg_w}} ST")
        divider = "-" * len(header)
        table   = header + "\n" + divider + "\n"
        for player, bldgs, defending, asleep, loc_time, avail in table_data:
            st   = "DEF" if defending else "ATK"
            icon = "💤" if asleep else "🟢"
            slot_tag = f"({avail})" if avail > 0 else "   "
            table += (f"{icon} {loc_time} {slot_tag} "
                      f"{player:<{name_w}} {bldgs:<{bldg_w}} {st}\n")

    total = sum(ctx.players_dict.values())

    # ── Assemble report sequentially ──────────────────────────────────────────
    infos  = (f"```ex\n"
              f"⚔️  Opponent: {opponent}\n"
              f"{team_icon} Team:     {team_points:,} (+{team_bonus})"
              f"  {opp_icon} Opponent: {opp_points:,} (+{opp_bonus})\n"
              f"⏳ Time Left: {h:02d}H {m:02d}m  |  "
              f"📊 Projected: {team_points + (team_bonus * proj_mins):,} / {max_points:,}\n"
              f"```\n")

    infos += f"🕐 Current Time: <t:{now_ts}:t>\n"
    instant = False

    if (team_time_m > 0
            and (team_time_m < opp_time_m or not opp_bonus)
            and team_time_m <= timer_left):
        instant = True
        infos += f"✅ WIN — team instants: <t:{team_instant}:t>\n"
        infos += (f"```ex\n"
                  f"⏱️  Instant in: {_fmt_hm(team_time_m)} ({team_time_m}m) (bonus: +{team_bonus})\n"
                  f"{table}"
                  f"🤖 Total: {total}/{BOT_SLOTS_TOTAL} bots placed```")

    elif (opp_time_m > 0
          and (opp_time_m < team_time_m or not team_bonus)
          and opp_time_m <= timer_left):
        instant = True
        infos += f"⚠️  LOSS — opponent instants: <t:{opp_instant}:t>\n"
        infos += (f"```ex\n"
                  f"⏱️  Instant in: {_fmt_hm(opp_time_m)} ({opp_time_m}m) (bonus: +{opp_bonus})\n"
                  f"{table}"
                  f"🤖 Total: {total}/{BOT_SLOTS_TOTAL} bots placed```")

    else:
        infos += (f"```ex\n"
                  f"{table}"
                  f"🤖 Total: {total}/{BOT_SLOTS_TOTAL} bots placed```")

    # ── Building strength analysis (v5.0) ──────────────────────────────────
    from bot.analysis import (
        analyze_buildings, format_building_summary,
        format_top_players, get_momentum, format_momentum,
    )
    from bot.config import CFG as _cfg
    analysis = analyze_buildings(ctx.slot_results)
    alert_thresholds = _cfg.get("alert_thresholds", {})
    bldg_summary = format_building_summary(analysis, alert_thresholds)

    # Top players by average strength
    top_players_text = format_top_players(ctx.slot_results, limit=10)

    # Score momentum (requires war trajectory from DB)
    momentum_text = ""
    try:
        from bot.history import get_or_create_war, get_score_trajectory
        if opponent:
            war_id = get_or_create_war(opponent)
            trajectory = get_score_trajectory(war_id)
            if len(trajectory) >= 2:
                momentum = get_momentum(trajectory)
                momentum_text = format_momentum(momentum)
    except Exception as e:
        print(f"[report] Momentum analysis failed: {e}")

    # Append analysis sections
    if momentum_text:
        infos += f"\n📈 **Momentum:** {momentum_text}\n"
    if bldg_summary:
        infos += f"\n```\nBUILDING STRENGTH (averages)\n{bldg_summary}\n```\n"
    if top_players_text:
        infos += f"\n```\nTOP PLAYERS\n{top_players_text}\n```\n"

    meta = dict(
        instant=instant, opponent=opponent,
        team_points=team_points, opp_points=opp_points,
        team_bonus=team_bonus, opp_bonus=opp_bonus,
        max_points=max_points, h=h, m=m,
    )
    return infos, meta


# ── Player stats from build images ────────────────────────────────────────────
def _load_stats_cache(cache_path: str) -> dict:
    """Load the build stats cache from disk."""
    import json
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[report] Stats cache load failed: {e}")
    return {}


def _save_stats_cache(cache_path: str, cache: dict):
    """Save the build stats cache to disk."""
    import json
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"[stats] Failed to save cache: {e}")


def get_player_stats_from_builds() -> list:
    """Extract HP/ATK from saved build images in builds/ folder.

    Uses a JSON cache keyed by file modification time so only changed
    images are re-OCR'd. Opening the Stats dialog is instant on repeat views.

    Returns:
        List of dicts sorted by total_str descending:
        {player, total_hp, total_atk, total_str,
         hp_1, atk_1, hp_2, atk_2, hp_3, atk_3, cars_found}
    """
    from bot.config import BUILDS_DIR
    from bot.ocr import ocr_build_stats

    if not os.path.isdir(BUILDS_DIR):
        return []

    cache_path = os.path.join(BUILDS_DIR, "_stats_cache.json")
    cache = _load_stats_cache(cache_path)
    dirty = False

    results = []
    for player_dir in sorted(os.listdir(BUILDS_DIR)):
        pdir = os.path.join(BUILDS_DIR, player_dir)
        if not os.path.isdir(pdir):
            continue

        car_stats = []
        for i in range(1, 4):
            img_path = os.path.join(pdir, f"{player_dir}_{i}.PNG")
            if os.path.isfile(img_path):
                mtime = str(os.path.getmtime(img_path))
                cache_key = f"{player_dir}_{i}"
                cached = cache.get(cache_key)
                if cached and cached.get("mtime") == mtime:
                    hp, atk = cached["hp"], cached["atk"]
                else:
                    hp, atk = ocr_build_stats(img_path)
                    cache[cache_key] = {"mtime": mtime, "hp": hp, "atk": atk}
                    dirty = True
                car_stats.append((hp, atk))
            else:
                car_stats.append((0, 0))

        while len(car_stats) < 3:
            car_stats.append((0, 0))

        total_hp = sum(c[0] for c in car_stats)
        total_atk = sum(c[1] for c in car_stats)
        total_str = total_hp + total_atk
        cars_found = sum(1 for c in car_stats if c[0] > 0 or c[1] > 0)

        if cars_found > 0:
            results.append({
                "player": player_dir,
                "total_hp": total_hp,
                "total_atk": total_atk,
                "total_str": total_str,
                "hp_1": car_stats[0][0], "atk_1": car_stats[0][1],
                "hp_2": car_stats[1][0], "atk_2": car_stats[1][1],
                "hp_3": car_stats[2][0], "atk_3": car_stats[2][1],
                "cars_found": cars_found,
            })

    if dirty:
        _save_stats_cache(cache_path, cache)

    results.sort(key=lambda r: r["total_str"], reverse=True)
    return results


def format_player_stats_table(stats: list) -> str:
    """Format player stats into a readable table.

    Args:
        stats: list of dicts from get_player_stats_from_builds()

    Returns:
        Formatted string table sorted by total strength descending.
    """
    if not stats:
        return "No player builds found.\nRun a full scan to save build images."

    lines = []
    lines.append("PLAYER STATS (from build images)")
    lines.append("=" * 62)
    lines.append(f"{'PLAYER':<16} {'STRENGTH':>10} {'HP':>10} {'ATK':>10}  CARS")
    lines.append("-" * 62)

    grand_str = grand_hp = grand_atk = 0
    for row in stats:
        player    = row["player"]
        total_str = _fmt_stat(row["total_str"])
        total_hp  = _fmt_stat(row["total_hp"])
        total_atk = _fmt_stat(row["total_atk"])
        cars      = row.get("cars_found", 3)
        lines.append(f"{player:<16} {total_str:>10} {total_hp:>10} {total_atk:>10}  {cars}/3")
        grand_str += row["total_str"]
        grand_hp  += row["total_hp"]
        grand_atk += row["total_atk"]

    lines.append("-" * 62)
    lines.append(f"{'TOTAL':<16} {_fmt_stat(grand_str):>10} "
                 f"{_fmt_stat(grand_hp):>10} {_fmt_stat(grand_atk):>10}  "
                 f"{len(stats)} players")

    # Per-car breakdown
    lines.append("")
    lines.append("PER-CAR BREAKDOWN")
    lines.append("-" * 62)
    lines.append(f"{'PLAYER':<16} {'CAR 1':>13} {'CAR 2':>13} {'CAR 3':>13}")
    lines.append("-" * 62)
    for row in stats:
        c1_hp, c1_atk = row["hp_1"], row["atk_1"]
        c2_hp, c2_atk = row["hp_2"], row["atk_2"]
        c3_hp, c3_atk = row["hp_3"], row["atk_3"]
        c1 = _fmt_stat(c1_hp + c1_atk) if (c1_hp + c1_atk) > 0 else "---"
        c2 = _fmt_stat(c2_hp + c2_atk) if (c2_hp + c2_atk) > 0 else "---"
        c3 = _fmt_stat(c3_hp + c3_atk) if (c3_hp + c3_atk) > 0 else "---"
        lines.append(f"{row['player']:<16} {c1:>13} {c2:>13} {c3:>13}")

    return "\n".join(lines)


# ── Build image grid ─────────────────────────────────────────────────────────
def build_image_grid(screenshot_paths: list, cols: int = 3):
    """Compose a grid collage of build screenshots.

    Args:
        screenshot_paths: list of file paths to build PNG images
        cols: number of columns in the grid (default 3)

    Returns:
        PIL Image suitable for posting to Discord as attachment,
        or None if no valid images.
    """
    from PIL import Image as Img

    images = []
    for p in screenshot_paths:
        if os.path.isfile(p):
            try:
                images.append(Img.open(p).convert("RGB"))
            except Exception as e:
                print(f"[report] Failed to load image {p}: {e}")

    if not images:
        return None

    # Resize all to match the first image's dimensions
    cell_w, cell_h = images[0].size
    resized = []
    for img in images:
        if img.size != (cell_w, cell_h):
            img = img.resize((cell_w, cell_h), Img.LANCZOS)
        resized.append(img)

    rows = math.ceil(len(resized) / cols)
    grid = Img.new("RGB", (cell_w * cols, cell_h * rows), (255, 255, 255))

    for i, img in enumerate(resized):
        r, c = divmod(i, cols)
        grid.paste(img, (c * cell_w, r * cell_h))

    return grid
