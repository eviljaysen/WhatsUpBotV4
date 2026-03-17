"""bot/history.py — SQLite scan history, war tracking, and player stats (v5.0).

Persists every scan result so reports can include change-deltas, HP trends,
opponent history, player strength tracking, score trajectories, and war results.

DB is created at BASE_DIR/history.db on first call to init_db().
All reads/writes are safe to call from background scan threads.
"""

import sqlite3
import time
from contextlib import contextmanager

from bot.config import DB_PATH


# ── Schema ─────────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS scans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   INTEGER NOT NULL,
    opponent    TEXT,
    team_pts    INTEGER,
    opp_pts     INTEGER,
    team_bonus  INTEGER,
    opp_bonus   INTEGER,
    max_pts     INTEGER,
    timer_h     INTEGER,
    timer_m     INTEGER
);

CREATE TABLE IF NOT EXISTS slots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id     INTEGER NOT NULL REFERENCES scans(id),
    player      TEXT,
    building    INTEGER,
    hp          INTEGER,
    atk         INTEGER,
    defending   INTEGER,
    screenshot  TEXT
);

CREATE TABLE IF NOT EXISTS player_stats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id     INTEGER NOT NULL REFERENCES scans(id),
    timestamp   INTEGER NOT NULL,
    player      TEXT    NOT NULL,
    total_hp    INTEGER NOT NULL,
    total_atk   INTEGER NOT NULL,
    total_str   INTEGER NOT NULL,
    hp_1        INTEGER NOT NULL DEFAULT 0,
    atk_1       INTEGER NOT NULL DEFAULT 0,
    hp_2        INTEGER NOT NULL DEFAULT 0,
    atk_2       INTEGER NOT NULL DEFAULT 0,
    hp_3        INTEGER NOT NULL DEFAULT 0,
    atk_3       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS wars (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    opponent    TEXT    NOT NULL,
    start_ts    INTEGER NOT NULL,
    end_ts      INTEGER,
    result      TEXT    DEFAULT 'ongoing',
    our_final   INTEGER DEFAULT 0,
    opp_final   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS score_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id     INTEGER NOT NULL REFERENCES scans(id),
    war_id      INTEGER NOT NULL REFERENCES wars(id),
    timestamp   INTEGER NOT NULL,
    team_pts    INTEGER NOT NULL,
    opp_pts     INTEGER NOT NULL,
    team_bonus  INTEGER DEFAULT 0,
    opp_bonus   INTEGER DEFAULT 0,
    timer_h     INTEGER DEFAULT 0,
    timer_m     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS opponent_players (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    war_id      INTEGER REFERENCES wars(id),
    opponent    TEXT    NOT NULL,
    player_name TEXT    NOT NULL,
    building    INTEGER,
    hp          INTEGER DEFAULT 0,
    atk         INTEGER DEFAULT 0,
    screenshot  TEXT,
    timestamp   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scans_opponent    ON scans(opponent);
CREATE INDEX IF NOT EXISTS idx_scans_timestamp   ON scans(timestamp);
CREATE INDEX IF NOT EXISTS idx_slots_player      ON slots(player);
CREATE INDEX IF NOT EXISTS idx_pstats_player     ON player_stats(player);
CREATE INDEX IF NOT EXISTS idx_pstats_timestamp  ON player_stats(timestamp);
CREATE INDEX IF NOT EXISTS idx_wars_opponent     ON wars(opponent);
CREATE INDEX IF NOT EXISTS idx_wars_result       ON wars(result);
CREATE INDEX IF NOT EXISTS idx_snapshots_war     ON score_snapshots(war_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts      ON score_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_oppplayers_war    ON opponent_players(war_id);
CREATE INDEX IF NOT EXISTS idx_oppplayers_name   ON opponent_players(player_name);
"""


@contextmanager
def _conn():
    """Context manager: open DB, yield connection, auto-commit + close."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    """Create tables and indexes if they don't exist. Safe to call repeatedly."""
    with _conn() as con:
        con.executescript(_DDL)
    print(f"[history] DB ready: {DB_PATH}")


# ── Write ──────────────────────────────────────────────────────────────────────
def save_scan(meta: dict, slot_results: list) -> int:
    """Persist a completed scan to the database.

    Args:
        meta: dict from build_report() — opponent, team_points, opp_points, etc.
        slot_results: list of SlotData instances

    Returns:
        scan_id (int) for the newly created scan row
    """
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO scans (timestamp, opponent, team_pts, opp_pts, "
            "team_bonus, opp_bonus, max_pts, timer_h, timer_m) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (int(time.time()),
             meta.get("opponent", ""),
             meta.get("team_points", 0),
             meta.get("opp_points", 0),
             meta.get("team_bonus", 0),
             meta.get("opp_bonus", 0),
             meta.get("max_points", 0),
             meta.get("timer_h", 0),
             meta.get("timer_m", 0)))
        scan_id = cur.lastrowid

        for s in slot_results:
            con.execute(
                "INSERT INTO slots (scan_id, player, building, hp, atk, "
                "defending, screenshot) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (scan_id, s.player, s.building, s.hp, s.atk,
                 1 if s.defending else 0, s.screenshot))

    print(f"[history] Saved scan #{scan_id}: {len(slot_results)} slots")
    return scan_id


def save_player_stats(scan_id: int, timestamp: int,
                      player: str, slots: list):
    """Save aggregated strength stats for a player with all 3 cars.

    Args:
        scan_id: ID of the parent scan
        timestamp: Unix epoch
        player: canonical player name
        slots: list of 3 SlotData instances (sorted by building)
    """
    slots_sorted = sorted(slots, key=lambda s: s.building)
    total_hp = sum(s.hp for s in slots_sorted)
    total_atk = sum(s.atk for s in slots_sorted)
    total_str = total_hp + total_atk

    with _conn() as con:
        con.execute(
            "INSERT INTO player_stats "
            "(scan_id, timestamp, player, total_hp, total_atk, total_str, "
            " hp_1, atk_1, hp_2, atk_2, hp_3, atk_3) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (scan_id, timestamp, player, total_hp, total_atk, total_str,
             slots_sorted[0].hp, slots_sorted[0].atk,
             slots_sorted[1].hp, slots_sorted[1].atk,
             slots_sorted[2].hp, slots_sorted[2].atk))


# ── Read ───────────────────────────────────────────────────────────────────────
def get_latest_player_stats() -> list:
    """Return the most recent stats row per player.

    Returns:
        List of dicts with keys: player, total_hp, total_atk, total_str,
        hp_1..3, atk_1..3, timestamp. Sorted by total_str descending.
    """
    sql = """
        SELECT ps.*
        FROM player_stats ps
        INNER JOIN (
            SELECT player, MAX(timestamp) AS max_ts
            FROM player_stats
            GROUP BY player
        ) latest ON ps.player = latest.player AND ps.timestamp = latest.max_ts
        ORDER BY ps.total_str DESC
    """
    with _conn() as con:
        rows = con.execute(sql).fetchall()
        return [dict(r) for r in rows]


def get_player_stats_history(player: str, limit: int = 10) -> list:
    """Return the last N strength snapshots for a player (most recent first)."""
    sql = """
        SELECT * FROM player_stats
        WHERE player = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """
    with _conn() as con:
        rows = con.execute(sql, (player, limit)).fetchall()
        return [dict(r) for r in rows]


def get_last_scan(opponent: str) -> tuple:
    """Return (meta_dict, slots_list) for the most recent scan vs opponent.

    Returns (None, []) if no previous scan found.
    """
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM scans WHERE opponent = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (opponent,)).fetchone()
        if row is None:
            return None, []
        meta = dict(row)
        slots = con.execute(
            "SELECT * FROM slots WHERE scan_id = ?",
            (meta["id"],)).fetchall()
        return meta, [dict(s) for s in slots]


def get_player_hp_trend(player: str, limit: int = 5) -> list:
    """Return a list of the last N total HP values for a player (most recent first)."""
    sql = """
        SELECT total_hp FROM player_stats
        WHERE player = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """
    with _conn() as con:
        rows = con.execute(sql, (player, limit)).fetchall()
        return [r["total_hp"] for r in rows]


def get_opponent_history(opponent: str, limit: int = 10) -> list:
    """Return a list of scan meta dicts for an opponent (most recent first)."""
    sql = """
        SELECT * FROM scans
        WHERE opponent = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """
    with _conn() as con:
        rows = con.execute(sql, (opponent, limit)).fetchall()
        return [dict(r) for r in rows]


# ── War tracking (v5.0) ─────────────────────────────────────────────────────
def get_or_create_war(opponent: str) -> int:
    """Find an ongoing war vs opponent, or create a new one.

    A war is considered "ongoing" if result == 'ongoing'.
    Returns the war_id.
    """
    with _conn() as con:
        row = con.execute(
            "SELECT id FROM wars WHERE opponent = ? AND result = 'ongoing' "
            "ORDER BY start_ts DESC LIMIT 1",
            (opponent,)).fetchone()
        if row is not None:
            return row["id"]
        cur = con.execute(
            "INSERT INTO wars (opponent, start_ts) VALUES (?, ?)",
            (opponent, int(time.time())))
        war_id = cur.lastrowid
        print(f"[history] New war #{war_id} vs {opponent}")
        return war_id


def save_score_snapshot(scan_id: int, war_id: int, meta: dict):
    """Save a point-in-time score snapshot for a war.

    Args:
        scan_id: ID of the parent scan
        war_id: ID of the war
        meta: dict with team_points, opp_points, team_bonus, opp_bonus,
              timer_h, timer_m
    """
    with _conn() as con:
        con.execute(
            "INSERT INTO score_snapshots "
            "(scan_id, war_id, timestamp, team_pts, opp_pts, "
            " team_bonus, opp_bonus, timer_h, timer_m) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (scan_id, war_id, int(time.time()),
             meta.get("team_points", 0),
             meta.get("opp_points", 0),
             meta.get("team_bonus", 0),
             meta.get("opp_bonus", 0),
             meta.get("timer_h", 0),
             meta.get("timer_m", 0)))


def get_score_trajectory(war_id: int) -> list:
    """Return all score snapshots for a war, ordered by time ascending."""
    sql = """
        SELECT * FROM score_snapshots
        WHERE war_id = ?
        ORDER BY timestamp ASC
    """
    with _conn() as con:
        rows = con.execute(sql, (war_id,)).fetchall()
        return [dict(r) for r in rows]


def close_war(war_id: int, result: str,
              our_final: int = 0, opp_final: int = 0):
    """Mark a war as complete.

    Args:
        war_id: ID of the war to close
        result: 'win', 'loss', or 'draw'
        our_final: our final score
        opp_final: opponent final score
    """
    with _conn() as con:
        con.execute(
            "UPDATE wars SET end_ts = ?, result = ?, "
            "our_final = ?, opp_final = ? WHERE id = ?",
            (int(time.time()), result, our_final, opp_final, war_id))
    print(f"[history] War #{war_id} closed: {result} ({our_final}-{opp_final})")


def save_enemy_slot(war_id: int, opponent: str, player_name: str,
                    building: int, hp: int = 0, atk: int = 0,
                    screenshot: str = "", timestamp: int = 0):
    """Save a scouted enemy player slot.

    Args:
        war_id: ID of the current war (or None if no war context)
        opponent: opponent team name
        player_name: enemy player name
        building: building number
        hp: HP value (0 if unknown)
        atk: ATK value (0 if unknown)
        screenshot: path to screenshot file
        timestamp: Unix epoch (0 = use current time)
    """
    ts = timestamp or int(time.time())
    with _conn() as con:
        con.execute(
            "INSERT INTO opponent_players "
            "(war_id, opponent, player_name, building, hp, atk, "
            " screenshot, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (war_id, opponent, player_name, building, hp, atk,
             screenshot, ts))


def get_opponent_roster(opponent: str) -> list:
    """Return all known enemy players across wars vs this opponent.

    Returns the most recent entry per player_name, sorted by total
    strength (hp + atk) descending.
    """
    sql = """
        SELECT op.*
        FROM opponent_players op
        INNER JOIN (
            SELECT player_name, MAX(timestamp) AS max_ts
            FROM opponent_players
            WHERE opponent = ?
            GROUP BY player_name
        ) latest ON op.player_name = latest.player_name
                 AND op.timestamp = latest.max_ts
        WHERE op.opponent = ?
        ORDER BY (op.hp + op.atk) DESC
    """
    with _conn() as con:
        rows = con.execute(sql, (opponent, opponent)).fetchall()
        return [dict(r) for r in rows]


def get_war_history(limit: int = 20) -> list:
    """Return recent wars with results (most recent first)."""
    sql = """
        SELECT * FROM wars
        ORDER BY start_ts DESC
        LIMIT ?
    """
    with _conn() as con:
        rows = con.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_war_by_id(war_id: int) -> dict | None:
    """Return a single war record by ID, or None."""
    with _conn() as con:
        row = con.execute("SELECT * FROM wars WHERE id = ?",
                          (war_id,)).fetchone()
        return dict(row) if row else None
