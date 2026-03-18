"""WhatsUpBot v5.0 — WOLF PACK DEFENSIVE INTELLIGENCE.

Scans a BlueStacks game window via screenshot and OCR to generate formatted
defense reports. Identifies players by name across
team and enemy buildings, captures HP/ATK/score/timer values, and produces
a Discord-ready report with player stats, win projection, and bot availability.

Entry point and tkinter UI only. All scan logic is in the bot/ package.

UI runs on the main thread. Scans run on daemon background threads.
All tkinter mutations from threads use root.after(0, lambda m=msg: fn(m)).
"""

VERSION = "5.0.0"

import os
import re
import sys
import threading

import tkinter as tk
from tkinter import Text, WORD

import pytz
from datetime import datetime

from bot.config import SCANS_DIR, IMAGES_DIR
from bot.scan import run_team_scan, run_enemy_scan
from bot.discord_post import post_report, DISCORD_WEBHOOK_URL

# System tray — optional dependency
try:
    import pystray
    from PIL import Image as PilImg
    _TRAY_AVAILABLE = True
except ImportError:
    _TRAY_AVAILABLE = False
    print("[tray] pystray not installed — system tray disabled")


# ── Crash handler ──────────────────────────────────────────────────────────────
def _crash_handler(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    import traceback
    from bot.config import BASE_DIR
    log_path = os.path.join(BASE_DIR, "crash.log")
    tb_str   = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"WhatsUpBot {VERSION} crash\n{'='*60}\n{tb_str}")
    print(f"\n{'='*60}\nCRASH: {exc_value}\nDetails: {log_path}\n{'='*60}")
    try:
        import tkinter.messagebox as mb
        mb.showerror("WhatsUpBot — Error",
                     f"{exc_value}\n\nFull details saved to:\n{log_path}")
    except Exception as e:
        print(f"[crash] Could not show error dialog: {e}")

sys.excepthook = _crash_handler


# ── Help text ──────────────────────────────────────────────────────────────────
def _load_help() -> str:
    from bot.config import BASE_DIR
    path = os.path.join(BASE_DIR, "README.txt")
    return open(path, encoding="utf-8").read() if os.path.isfile(path) else "README.txt not found."


# ── Main App ───────────────────────────────────────────────────────────────────
class App:
    """Persistent WhatsUpBot window. Scans run on background daemon threads."""

    def __init__(self, root: tk.Tk):
        self.root          = root
        self._scan_thread  = None
        self._last_meta    = {}
        self._interval_job = None   # auto-scan job ID
        self._tray_icon    = None   # pystray Icon instance
        self._build_ui()
        self._setup_tray()
        # Wire correction dialog callback — fires on main thread via root.after
        # The active ScanContext is stored here so the dialog can signal it
        self._active_scan_ctx = None
        self._correction_cb = lambda raw: self.root.after(
            0, lambda r=raw: self._show_correction_dialog(r))

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        r = self.root
        r.configure(bg="white")

        tk.Label(r, text=f"WhatsUp Bot — Wolf Pack Defense  v{VERSION}",
                 font=("Helvetica", 14, "bold"), bg="white").pack(pady=10)

        btn_row = tk.Frame(r, bg="white")
        btn_row.pack(pady=4)

        self.btn_team = tk.Button(
            btn_row, text="Team Scan",
            command=self._on_team_scan,
            bg="#4CAF50", fg="white", font=("Helvetica", 11, "bold"),
            relief="flat", padx=10, pady=5, width=13)
        self.btn_team.pack(side="left", padx=8)

        self.btn_enemy = tk.Button(
            btn_row, text="Enemy Scan",
            command=self._on_enemy_scan,
            bg="#E53935", fg="white", font=("Helvetica", 11, "bold"),
            relief="flat", padx=10, pady=5, width=13)
        self.btn_enemy.pack(side="left", padx=8)

        # Enemy building selector
        sel_row = tk.Frame(r, bg="white")
        sel_row.pack(pady=2)
        tk.Label(sel_row, text="Enemy building:", bg="white").pack(side="left")
        self.building_var = tk.IntVar(value=1)
        tk.Spinbox(sel_row, from_=1, to=6, width=3,
                   textvariable=self.building_var,
                   font=("Helvetica", 10)).pack(side="left", padx=4)

        # v4.0: auto-scan interval row
        int_row = tk.Frame(r, bg="white")
        int_row.pack(pady=2)
        tk.Label(int_row, text="Auto-scan every (min, 0=off):", bg="white").pack(side="left")
        self.interval_var = tk.IntVar(value=0)
        tk.Spinbox(int_row, from_=0, to=120, width=4,
                   textvariable=self.interval_var,
                   font=("Helvetica", 10)).pack(side="left", padx=4)
        tk.Button(int_row, text="Start", command=self._toggle_interval,
                  width=6).pack(side="left", padx=4)

        # Toggle: show only players with available slots
        avail_row = tk.Frame(r, bg="white")
        avail_row.pack(pady=2)
        self.avail_only_var = tk.BooleanVar(value=True)
        tk.Checkbutton(avail_row, text="Show only available players",
                       variable=self.avail_only_var, bg="white",
                       font=("Helvetica", 9)).pack()

        self.status_var = tk.StringVar(value="Ready — press a scan button to begin.")
        tk.Label(r, textvariable=self.status_var, fg="#555555", bg="white",
                 font=("Helvetica", 9, "italic")).pack(pady=4)

        txt_frame = tk.Frame(r)
        txt_frame.pack(padx=10, pady=4, fill="both", expand=True)
        sb = tk.Scrollbar(txt_frame)
        sb.pack(side="right", fill="y")
        self.txt = Text(txt_frame, wrap=WORD, yscrollcommand=sb.set,
                        font=("Consolas", 9), height=18)
        self.txt.pack(side="left", fill="both", expand=True)
        sb.config(command=self.txt.yview)

        # ── v5.0: Building strength mini-display ────────────────────────────
        self._bldg_frame = tk.Frame(r, bg="white")
        self._bldg_frame.pack(padx=10, fill="x")
        self._bldg_text = tk.Label(
            self._bldg_frame, text="", font=("Consolas", 8),
            bg="white", fg="#333", justify="left", anchor="w")
        self._bldg_text.pack(fill="x")

        bot_row = tk.Frame(r, bg="white")
        bot_row.pack(pady=8)
        for label, cmd in [("Copy",     self._copy),
                            ("Save",     self._save),
                            ("Post",     self._post_discord),
                            ("Stats",    self._show_player_stats),
                            ("Scouting", self._show_scouting),
                            ("War",      self._show_war_dialog),
                            ("HUD",      self._calibrate_hud),
                            ("Help",     self._help),
                            ("Quit",     self._quit)]:
            tk.Button(bot_row, text=label, command=cmd, width=8).pack(side="left", padx=3)

    # ── Thread helpers ─────────────────────────────────────────────────────────
    def _ui(self, fn):
        self.root.after(0, fn)

    def _set_buttons(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_team.config(state=state)
        self.btn_enemy.config(state=state)

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def _set_results(self, text: str):
        self.txt.delete("1.0", tk.END)
        self.txt.insert("1.0", text)

    def _status_cb(self, msg: str):
        self._ui(lambda m=msg: self._set_status(m))

    def _launch_scan(self, worker):
        if self._scan_thread and self._scan_thread.is_alive():
            return
        self._set_buttons(False)
        self._scan_thread = threading.Thread(target=worker, daemon=True)
        self._scan_thread.start()

    # ── Scan launchers ─────────────────────────────────────────────────────────
    def _on_team_scan(self):
        self._launch_scan(self._team_worker)

    def _on_enemy_scan(self):
        building_num = self.building_var.get()
        self._launch_scan(lambda: self._enemy_worker(building_num))

    # ── Scan workers ───────────────────────────────────────────────────────────
    def _team_worker(self):
        try:
            infos, meta = run_team_scan(
                status_cb=self._status_cb,
                correction_cb=self._correction_cb,
                avail_only=self.avail_only_var.get(),
            )
            self._last_meta = meta
            self._ui(lambda t=infos: self._set_results(t))
            self._ui(lambda: self._set_status("Team scan complete."))
            opp = meta.get("opponent", "")
            tp = meta.get("team_points", 0)
            op = meta.get("opp_points", 0)
            self._tray_notify("Team Scan Complete",
                              f"vs {opp}: {tp:,} - {op:,}")
            # v5.0: update building strength display + war status
            self._ui(lambda: self._update_building_display())
            self._ui(lambda o=opp: self._update_war_status(o))
        except Exception as exc:
            self._ui(lambda e=exc: self._set_results(f"[ERROR] {e}"))
            self._ui(lambda: self._set_status("Scan error — see results."))
            raise
        finally:
            self._ui(lambda: self._set_buttons(True))

    def _enemy_worker(self, building_num: int):
        try:
            summary = run_enemy_scan(building_num, status_cb=self._status_cb)
            self._last_meta = {"scan_type": "enemy", "building": building_num}
            self._ui(lambda t=summary: self._set_results(t))
            self._ui(lambda: self._set_status(
                f"Enemy scan — building {building_num} complete."))
            self._tray_notify("Enemy Scan Complete",
                              f"Building {building_num} scanned.")
        except Exception as exc:
            self._ui(lambda e=exc: self._set_results(f"[ERROR] {e}"))
            self._ui(lambda: self._set_status("Scan error — see results."))
            raise
        finally:
            self._ui(lambda: self._set_buttons(True))

    # ── v4.0: Auto-scan interval ───────────────────────────────────────────────
    def _toggle_interval(self):
        """Start or stop the auto-scan interval."""
        if self._interval_job is not None:
            self.root.after_cancel(self._interval_job)
            self._interval_job = None
            self._set_status("Auto-scan stopped.")
            return

        minutes = self.interval_var.get()
        if minutes <= 0:
            self._set_status("Set interval > 0 to enable auto-scan.")
            return

        ms = minutes * 60 * 1000

        def _fire():
            self._on_team_scan()
            self._interval_job = self.root.after(ms, _fire)

        self._interval_job = self.root.after(ms, _fire)
        self._set_status(f"Auto-scan every {minutes}m. Next in {minutes}m.")

    # ── Bottom button handlers ─────────────────────────────────────────────────
    def _copy(self):
        text = self.txt.get("1.0", tk.END).strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("Copied to clipboard.")

    def _save(self):
        body = self.txt.get("1.0", tk.END).strip()
        if not body:
            self._set_status("Nothing to save yet.")
            return
        meta     = self._last_meta
        ts       = datetime.now(pytz.utc).strftime("%d-%m-%Y_%Hh%M utc")
        opponent = meta.get("opponent", "unknown")
        safe_opp = re.sub(r'[\\/:*?"<>|]', '', opponent)
        path     = os.path.join(SCANS_DIR, f"{safe_opp} ; {ts}.txt")

        if meta.get("instant"):
            header = (f"vs {opponent}\n"
                      f"team: {meta['team_points']:,} (+{meta['team_bonus']})"
                      f"  opp: {meta['opp_points']:,} (+{meta['opp_bonus']})\n"
                      f"time: {datetime.now().strftime('%H:%M')}"
                      f" | {meta['h']:02d}H {meta['m']:02d}m left\n"
                      f"score: {meta['team_points']:,} / {meta['max_points']:,}\n")
            body = header + body

        body = body.replace("```ex", "").replace("```", "")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(body)
        self._set_status(f"Saved: {os.path.basename(path)}")
        print(f"Report saved: {path}")

    def _post_discord(self):
        """v4.0: Post current report text to Discord webhook."""
        text = self.txt.get("1.0", tk.END).strip()
        if not text:
            self._set_status("Nothing to post.")
            return
        if not DISCORD_WEBHOOK_URL:
            self._set_status("No discord_webhook_url in config.json.")
            return

        def _worker():
            ok = post_report(text)
            msg = "Posted to Discord." if ok else "Discord post failed — check console."
            self._ui(lambda m=msg: self._set_status(m))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_correction_dialog(self, raw: str):
        """Main-thread dialog: let user resolve an unrecognized OCR name."""
        from bot.config import _KNOWN_NAMES
        from bot.ocr import _name_match
        from bot.config import save_ocr_correction

        dialog = tk.Toplevel(self.root)
        dialog.title("Resolve OCR Name")
        dialog.configure(bg="white")
        dialog.grab_set()
        dialog.lift()

        tk.Label(dialog, text=f'Unrecognized OCR result:  "{raw}"',
                 font=("Helvetica", 10, "bold"), bg="white").pack(pady=(12, 4), padx=16)
        tk.Label(dialog, text="Select the correct player or type a custom name:",
                 bg="white").pack(padx=16)

        list_frame = tk.Frame(dialog)
        list_frame.pack(padx=16, pady=6, fill="both", expand=True)
        sb = tk.Scrollbar(list_frame)
        sb.pack(side="right", fill="y")
        lb = tk.Listbox(list_frame, selectmode=tk.SINGLE, height=10,
                        yscrollcommand=sb.set, exportselection=False)
        for name in sorted(_KNOWN_NAMES):
            lb.insert(tk.END, name)
        lb.pack(side="left", fill="both", expand=True)
        sb.config(command=lb.yview)

        entry_var = tk.StringVar()
        tk.Label(dialog, text="Custom name:", bg="white").pack(padx=16)
        tk.Entry(dialog, textvariable=entry_var, width=28,
                 font=("Consolas", 10)).pack(padx=16, pady=4)

        save_var = tk.BooleanVar(value=True)
        tk.Checkbutton(dialog, text="Save as OCR correction (persist to config.json)",
                       variable=save_var, bg="white").pack(padx=16)

        # We need access to the active ScanContext for correction_event/result.
        # The active context is threaded through correction_cb — this dialog
        # sets the result and signals the event that ocr_player_name() is waiting on.
        # The App holds a reference to the last ScanContext via the correction_cb closure.
        _result_holder = [None]
        _event_holder  = [None]

        def _confirm():
            sel  = lb.curselection()
            name = lb.get(sel[0]) if sel else entry_var.get().strip()
            if not name:
                name = raw
            if save_var.get() and name and name.upper() != raw.upper():
                save_ocr_correction(raw.upper(), name)
                # Also save name template if last_name_shot available
                # (correction_cb closure provides name_matcher reference)
            _result_holder[0] = name
            dialog.destroy()

        def _skip():
            _result_holder[0] = None
            dialog.destroy()

        btn = tk.Frame(dialog, bg="white")
        btn.pack(pady=10)
        tk.Button(btn, text="Confirm", command=_confirm,
                  bg="#4CAF50", fg="white", width=10).pack(side="left", padx=6)
        tk.Button(btn, text="Skip", command=_skip, width=8).pack(side="left", padx=6)
        dialog.protocol("WM_DELETE_WINDOW", _skip)

        # Block until dialog closes, then signal the waiting scan thread
        dialog.wait_window()

        # The correction_cb wrapper in ScanContext.create() carries a _ctx ref.
        # Use it to push the result back and unblock ocr_player_name().
        cb = self._correction_cb
        ctx = getattr(cb, '_ctx', None)
        if ctx is not None:
            ctx.correction_result[0] = _result_holder[0]
            ctx.correction_event.set()

    def _show_player_stats(self):
        """Open a dialog showing player strength stats from build images."""
        win = tk.Toplevel(self.root)
        win.title("Player Stats")
        win.configure(bg="white")

        status_label = tk.Label(win, text="Extracting stats from build images...",
                                fg="#555", bg="white", font=("Helvetica", 9, "italic"))
        status_label.pack(pady=4)

        txt = Text(win, wrap=WORD, width=72, height=30, bg="white", relief="flat",
                   font=("Consolas", 9))
        txt.pack(padx=16, pady=10, fill="both", expand=True)
        txt.insert("1.0", "Loading...")
        txt.config(state="disabled")

        btn_row = tk.Frame(win, bg="white")
        btn_row.pack(pady=8)

        def _load_stats():
            try:
                from bot.report import get_player_stats_from_builds, format_player_stats_table
                stats = get_player_stats_from_builds()
                table = format_player_stats_table(stats)
            except Exception as e:
                table = f"Error loading stats: {e}"
            self._ui(lambda t=table: _update_text(t))

        def _update_text(table):
            txt.config(state="normal")
            txt.delete("1.0", "end")
            txt.insert("1.0", table)
            txt.config(state="disabled")
            status_label.config(text="Done")

        def _refresh():
            status_label.config(text="Refreshing...")
            threading.Thread(target=_load_stats, daemon=True).start()

        tk.Button(btn_row, text="Refresh", command=_refresh, width=8).pack(side="left", padx=4)
        tk.Button(btn_row, text="Close", command=win.destroy, width=8).pack(side="left", padx=4)

        # Load stats on background thread (OCR can be slow)
        threading.Thread(target=_load_stats, daemon=True).start()

    # ── v5.0: Building strength mini-display ─────────────────────────────────
    def _update_building_display(self):
        """Update the building strength summary below the report text."""
        try:
            from bot.report import get_player_stats_from_builds
            from bot.analysis import analyze_buildings
            from bot.report import _fmt_stat
            stats = get_player_stats_from_builds()
            if not stats:
                self._bldg_text.config(text="")
                return
            # Build mock SlotData-like objects for analysis
            from types import SimpleNamespace
            slots = []
            for s in stats:
                for i in range(1, 4):
                    hp = s.get(f"hp_{i}", 0)
                    atk = s.get(f"atk_{i}", 0)
                    if hp > 0 or atk > 0:
                        slots.append(SimpleNamespace(
                            player=s["player"], building=i,
                            hp=hp, atk=atk, defending=True))
            if not slots:
                self._bldg_text.config(text="")
                return
            analysis = analyze_buildings(slots)
            parts = []
            for bnum in sorted(analysis["buildings"].keys()):
                b = analysis["buildings"][bnum]
                n = max(b["slot_count"], 1)
                avg_str = b["total_str"] // n
                flag = " ⚠" if bnum == analysis["weakest"] else ""
                parts.append(f"B{bnum}{flag}: avg {_fmt_stat(avg_str)} "
                             f"({b['slot_count']})")
            self._bldg_text.config(text="  ".join(parts))
        except Exception as e:
            print(f"[ui] Building display error: {e}")

    # ── v5.0: War status indicator ────────────────────────────────────────────
    def _update_war_status(self, opponent: str = ""):
        """Update the window title with war status."""
        try:
            if not opponent:
                self.root.title(f"Wolf Pack Defense v{VERSION}")
                return
            from bot.history import init_db, get_or_create_war, get_war_by_id
            init_db()
            war_id = get_or_create_war(opponent)
            war = get_war_by_id(war_id)
            if war and war["result"] == "ongoing":
                self.root.title(
                    f"Wolf Pack Defense v{VERSION} — ⚔ vs {opponent}")
            elif war:
                result_icon = {"win": "✅", "loss": "❌", "draw": "🟰"}.get(
                    war["result"], "")
                self.root.title(
                    f"Wolf Pack Defense v{VERSION} — {result_icon} vs {opponent}")
        except Exception as e:
            print(f"[ui] War status error: {e}")

    # ── v5.0: Scouting dialog ─────────────────────────────────────────────────
    def _show_scouting(self):
        """Show opponent roster from the database."""
        win = tk.Toplevel(self.root)
        win.title("Opponent Scouting")
        win.configure(bg="white")
        win.geometry("600x500")

        top_row = tk.Frame(win, bg="white")
        top_row.pack(fill="x", padx=12, pady=8)
        tk.Label(top_row, text="Opponent:", bg="white",
                 font=("Helvetica", 10)).pack(side="left")
        opp_var = tk.StringVar(value=self._last_meta.get("opponent", ""))
        opp_entry = tk.Entry(top_row, textvariable=opp_var, width=24,
                             font=("Consolas", 10))
        opp_entry.pack(side="left", padx=6)
        tk.Button(top_row, text="Load", command=lambda: _load(),
                  width=8).pack(side="left", padx=4)

        txt = Text(win, wrap=WORD, width=70, height=25, bg="white",
                   relief="flat", font=("Consolas", 9))
        txt.pack(padx=12, pady=6, fill="both", expand=True)
        txt.insert("1.0", "Enter an opponent name and click Load.")
        txt.config(state="disabled")

        btn_row = tk.Frame(win, bg="white")
        btn_row.pack(pady=8)
        tk.Button(btn_row, text="Close", command=win.destroy, width=8).pack()

        def _load():
            opponent = opp_var.get().strip()
            if not opponent:
                return

            def _worker():
                try:
                    from bot.history import (
                        init_db, get_opponent_roster, get_war_history,
                    )
                    from bot.report import _fmt_stat
                    init_db()

                    roster = get_opponent_roster(opponent)
                    wars = [w for w in get_war_history(50)
                            if w["opponent"] == opponent]

                    lines = [f"SCOUTING: {opponent}", "=" * 50]

                    if wars:
                        lines.append(f"\nWAR HISTORY ({len(wars)} wars)")
                        lines.append("-" * 50)
                        for w in wars:
                            result = w["result"].upper()
                            score = (f"{w['our_final']:,}-{w['opp_final']:,}"
                                     if w["our_final"] or w["opp_final"]
                                     else "")
                            lines.append(f"  {result:<8} {score}")
                    else:
                        lines.append("\nNo war history found.")

                    if roster:
                        lines.append(f"\nKNOWN PLAYERS ({len(roster)})")
                        lines.append("-" * 50)
                        lines.append(f"  {'PLAYER':<20} {'HP':>8} {'ATK':>8} "
                                     f"{'STR':>8}  BLDG")
                        lines.append("  " + "-" * 48)
                        for r in roster:
                            hp = r.get("hp", 0)
                            atk = r.get("atk", 0)
                            bldg = r.get("building", "?")
                            lines.append(
                                f"  {r['player_name']:<20} "
                                f"{_fmt_stat(hp):>8} "
                                f"{_fmt_stat(atk):>8} "
                                f"{_fmt_stat(hp + atk):>8}  "
                                f"B{bldg}")
                    else:
                        lines.append("\nNo scouted players found.")

                    result_text = "\n".join(lines)
                except Exception as e:
                    result_text = f"Error: {e}"
                self._ui(lambda t=result_text: _update(t))

            def _update(text):
                txt.config(state="normal")
                txt.delete("1.0", "end")
                txt.insert("1.0", text)
                txt.config(state="disabled")

            threading.Thread(target=_worker, daemon=True).start()

    # ── v5.0: War management dialog ──────────────────────────────────────────
    def _show_war_dialog(self):
        """Show war management dialog: view wars, close ongoing wars."""
        win = tk.Toplevel(self.root)
        win.title("War Management")
        win.configure(bg="white")
        win.geometry("550x450")

        txt = Text(win, wrap=WORD, width=65, height=18, bg="white",
                   relief="flat", font=("Consolas", 9))
        txt.pack(padx=12, pady=8, fill="both", expand=True)

        # Close-war controls
        close_frame = tk.Frame(win, bg="white")
        close_frame.pack(fill="x", padx=12, pady=4)

        tk.Label(close_frame, text="Close war ID:", bg="white").pack(side="left")
        war_id_var = tk.IntVar(value=0)
        tk.Spinbox(close_frame, from_=0, to=9999, width=5,
                   textvariable=war_id_var,
                   font=("Helvetica", 10)).pack(side="left", padx=4)

        result_var = tk.StringVar(value="win")
        for val in ["win", "loss", "draw"]:
            tk.Radiobutton(close_frame, text=val.title(), variable=result_var,
                           value=val, bg="white").pack(side="left", padx=2)

        tk.Label(close_frame, text="Our:", bg="white").pack(side="left", padx=(8, 2))
        our_var = tk.IntVar(value=0)
        tk.Entry(close_frame, textvariable=our_var, width=7,
                 font=("Consolas", 9)).pack(side="left")
        tk.Label(close_frame, text="Opp:", bg="white").pack(side="left", padx=(4, 2))
        opp_var = tk.IntVar(value=0)
        tk.Entry(close_frame, textvariable=opp_var, width=7,
                 font=("Consolas", 9)).pack(side="left")

        btn_row = tk.Frame(win, bg="white")
        btn_row.pack(pady=8)
        tk.Button(btn_row, text="Close War", command=lambda: _close_war(),
                  bg="#E53935", fg="white", width=10).pack(side="left", padx=4)
        tk.Button(btn_row, text="Refresh", command=lambda: _load(),
                  width=8).pack(side="left", padx=4)
        tk.Button(btn_row, text="Done", command=win.destroy,
                  width=8).pack(side="left", padx=4)

        def _load():
            def _worker():
                try:
                    from bot.history import init_db, get_war_history
                    from datetime import datetime as dt
                    init_db()
                    wars = get_war_history(limit=20)
                    lines = ["RECENT WARS", "=" * 52,
                             f"  {'ID':<5} {'OPPONENT':<18} {'RESULT':<10} "
                             f"{'SCORE':<15} DATE",
                             "  " + "-" * 50]
                    for w in wars:
                        score = (f"{w['our_final']:,}-{w['opp_final']:,}"
                                 if w["our_final"] or w["opp_final"] else "—")
                        date = dt.fromtimestamp(w["start_ts"]).strftime(
                            "%Y-%m-%d") if w["start_ts"] else "—"
                        lines.append(
                            f"  {w['id']:<5} {w['opponent']:<18} "
                            f"{w['result'].upper():<10} {score:<15} {date}")
                    if not wars:
                        lines.append("  No wars found.")
                    result_text = "\n".join(lines)
                except Exception as e:
                    result_text = f"Error: {e}"
                self._ui(lambda t=result_text: _update(t))

            def _update(text):
                txt.config(state="normal")
                txt.delete("1.0", "end")
                txt.insert("1.0", text)
                txt.config(state="disabled")

            threading.Thread(target=_worker, daemon=True).start()

        def _close_war():
            wid = war_id_var.get()
            if wid <= 0:
                return
            result = result_var.get()
            our = our_var.get()
            opp = opp_var.get()

            def _worker():
                try:
                    from bot.history import init_db, close_war
                    init_db()
                    close_war(wid, result, our, opp)
                    self._ui(lambda: self._set_status(
                        f"War #{wid} closed as {result}"))
                except Exception as e:
                    self._ui(lambda e=e: self._set_status(f"Error: {e}"))
                self._ui(_load)

            threading.Thread(target=_worker, daemon=True).start()

        _load()

    # ── HUD Calibration ─────────────────────────────────────────────────────
    def _calibrate_hud(self):
        """Open the HUD calibration dialog.

        Takes a screenshot of the game window, displays it with HUD region
        rectangles overlaid. User can select a region and drag to reposition.
        Saves overrides to config.json.
        """
        from bot.window import detect_window
        from bot.templates import _HUD, _HUD_DEFAULTS
        from bot.config import save_hud_overrides, CFG

        try:
            win = detect_window()
        except Exception as e:
            self._set_status(f"HUD calibration failed: {e}")
            return

        import pyautogui
        shot = pyautogui.screenshot(region=win.region)

        # Scale to fit in dialog (max 960 wide)
        scale = min(960 / shot.width, 720 / shot.height, 1.0)
        disp_w = int(shot.width * scale)
        disp_h = int(shot.height * scale)
        from PIL import Image as PilImg, ImageTk
        disp_img = shot.resize((disp_w, disp_h), PilImg.LANCZOS)

        dialog = tk.Toplevel(self.root)
        dialog.title("HUD Calibration")
        dialog.configure(bg="white")
        dialog.geometry(f"{disp_w + 200}x{disp_h + 80}")

        # Region selector
        top_row = tk.Frame(dialog, bg="white")
        top_row.pack(fill="x", padx=8, pady=4)

        tk.Label(top_row, text="Region:", bg="white").pack(side="left")
        region_var = tk.StringVar(value=list(_HUD.keys())[0])
        region_menu = tk.OptionMenu(top_row, region_var, *_HUD.keys())
        region_menu.pack(side="left", padx=4)

        coord_var = tk.StringVar(value="")
        tk.Label(top_row, textvariable=coord_var, bg="white",
                 font=("Consolas", 9)).pack(side="left", padx=8)

        tk.Button(top_row, text="Reset All", command=lambda: _reset_all()).pack(side="right", padx=4)
        tk.Button(top_row, text="Save", command=lambda: _save(),
                  bg="#4CAF50", fg="white").pack(side="right", padx=4)

        # Canvas with screenshot
        canvas = tk.Canvas(dialog, width=disp_w, height=disp_h,
                           bg="black", highlightthickness=0)
        canvas.pack(padx=8, pady=4)

        tk_img = ImageTk.PhotoImage(disp_img)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas._tk_img = tk_img  # prevent GC

        # Track overrides (start from current _HUD which includes any existing overrides)
        overrides = dict(CFG.get("hud_overrides", {}))
        # Convert stored lists to tuples
        overrides = {k: tuple(v) for k, v in overrides.items()}

        # Draw all regions
        rect_ids = {}

        def _bl_to_canvas(bx, by, bw, bh):
            """Convert 1920×1080 baseline to canvas pixel coords."""
            x1 = bx * win.sx * scale
            y1 = by * win.sy * scale
            x2 = (bx + bw) * win.sx * scale
            y2 = (by + bh) * win.sy * scale
            return x1, y1, x2, y2

        def _canvas_to_bl(x1, y1, x2, y2):
            """Convert canvas pixel coords back to 1920×1080 baseline."""
            bx = x1 / (win.sx * scale)
            by = y1 / (win.sy * scale)
            bw = (x2 - x1) / (win.sx * scale)
            bh = (y2 - y1) / (win.sy * scale)
            return (round(bx), round(by), round(bw), round(bh))

        def _draw_regions():
            for rid in rect_ids.values():
                canvas.delete(rid)
            rect_ids.clear()
            selected = region_var.get()
            for name, coords in _HUD.items():
                x1, y1, x2, y2 = _bl_to_canvas(*coords)
                color = "#00FF00" if name == selected else "#FFFF00"
                width = 2 if name == selected else 1
                rid = canvas.create_rectangle(x1, y1, x2, y2,
                                              outline=color, width=width)
                rect_ids[name] = rid
            bl = _HUD[selected]
            coord_var.set(f"({bl[0]}, {bl[1]}, {bl[2]}, {bl[3]})")

        _draw_regions()

        # Redraw on region change
        def _on_region_change(*_):
            _draw_regions()
        region_var.trace_add("write", _on_region_change)

        # Drag handling
        drag_state = {"start": None, "rect": None}

        def _on_press(event):
            drag_state["start"] = (event.x, event.y)
            if drag_state["rect"]:
                canvas.delete(drag_state["rect"])
            drag_state["rect"] = None

        def _on_drag(event):
            if drag_state["start"] is None:
                return
            sx, sy = drag_state["start"]
            if drag_state["rect"]:
                canvas.delete(drag_state["rect"])
            drag_state["rect"] = canvas.create_rectangle(
                sx, sy, event.x, event.y, outline="#FF0000", width=2, dash=(4, 2))

        def _on_release(event):
            if drag_state["start"] is None:
                return
            sx, sy = drag_state["start"]
            ex, ey = event.x, event.y
            drag_state["start"] = None
            if drag_state["rect"]:
                canvas.delete(drag_state["rect"])
                drag_state["rect"] = None

            # Minimum drag size
            if abs(ex - sx) < 5 or abs(ey - sy) < 5:
                return

            x1, y1 = min(sx, ex), min(sy, ey)
            x2, y2 = max(sx, ex), max(sy, ey)
            bl = _canvas_to_bl(x1, y1, x2, y2)
            name = region_var.get()
            _HUD[name] = bl
            overrides[name] = bl
            _draw_regions()

        canvas.bind("<ButtonPress-1>", _on_press)
        canvas.bind("<B1-Motion>", _on_drag)
        canvas.bind("<ButtonRelease-1>", _on_release)

        def _save():
            save_hud_overrides(overrides)
            self._set_status(f"HUD overrides saved ({len(overrides)} regions).")
            dialog.destroy()

        def _reset_all():
            overrides.clear()
            _HUD.clear()
            _HUD.update(_HUD_DEFAULTS)
            save_hud_overrides({})
            _draw_regions()
            self._set_status("HUD regions reset to defaults.")

    def _help(self):
        win = tk.Toplevel(self.root)
        win.title("Help")
        win.configure(bg="white")
        txt = Text(win, wrap=WORD, width=72, height=32, bg="white", relief="flat",
                   font=("Consolas", 9))
        txt.pack(padx=16, pady=10, fill="both", expand=True)
        txt.insert("1.0", _load_help())
        txt.config(state="disabled")
        tk.Button(win, text="Close", command=win.destroy).pack(pady=8)

    # ── System tray ──────────────────────────────────────────────────────────
    def _setup_tray(self):
        """Create the system tray icon. Gracefully skipped if pystray unavailable."""
        if not _TRAY_AVAILABLE:
            return
        # Use custom icon if available, otherwise generate one
        icon_path = os.path.join(IMAGES_DIR, "tray_icon.png")
        if os.path.isfile(icon_path):
            icon_img = PilImg.open(icon_path).convert("RGBA")
        else:
            icon_img = self._generate_tray_icon()

        menu = pystray.Menu(
            pystray.MenuItem("Show", self._restore_from_tray, default=True),
            pystray.MenuItem("Team Scan", lambda: self.root.after(0, self._on_team_scan)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: self.root.after(0, self._quit)),
        )
        self._tray_icon = pystray.Icon("WhatsUpBot", icon_img,
                                        "WhatsUpBot", menu)
        threading.Thread(target=self._tray_icon.run, daemon=True).start()
        # Intercept window close → minimize to tray instead of quit
        self.root.protocol("WM_DELETE_WINDOW", self._minimize_to_tray)

    @staticmethod
    def _generate_tray_icon():
        """Generate a simple 64x64 tray icon with 'W' text."""
        from PIL import ImageDraw, ImageFont
        img = PilImg.new("RGBA", (64, 64), (76, 175, 80, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except Exception:
            font = ImageFont.load_default()
        draw.text((12, 8), "W", fill=(255, 255, 255, 255), font=font)
        return img

    def _minimize_to_tray(self):
        """Hide the main window to the system tray."""
        if self._tray_icon is not None:
            self.root.withdraw()
        else:
            self._quit()

    def _restore_from_tray(self, *_args):
        """Restore the main window from system tray."""
        self.root.after(0, self._do_restore)

    def _do_restore(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _tray_notify(self, title: str, message: str):
        """Show a system tray notification balloon."""
        if self._tray_icon is not None:
            try:
                self._tray_icon.notify(message, title)
            except Exception as e:
                print(f"[tray] Notification failed: {e}")

    def _quit(self):
        if self._interval_job:
            self.root.after_cancel(self._interval_job)
        if self._tray_icon is not None:
            try:
                self._tray_icon.stop()
            except Exception as e:
                print(f"[tray] Stop error: {e}")
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Wolf Pack Defense")
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    ww = max(480, sw  // 3)
    wh = max(640, sh * 2 // 3)
    root.geometry(f"{ww}x{wh}+{(sw - ww) // 2}+{(sh - wh) // 2}")
    App(root)
    root.mainloop()
