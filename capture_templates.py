"""capture_templates.py — Interactive template crop helper.

Opens the saved debug_window.PNG (or any screenshot you choose) and lets you
drag-select regions to save as the required template images.

Usage:
    python capture_templates.py
    python capture_templates.py path/to/screenshot.png
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

TEMPLATES = [
    ("lobby.PNG",     "Lobby screen (DEFEND / ATTACK choice — crop a distinctive area)"),
    ("team_logo.png", "Team logo icon on player card (left edge of card in slot view)"),
]


class CropSelector:
    def __init__(self, root: tk.Tk, image_path: str):
        self.root       = root
        self.src_image  = Image.open(image_path).convert("RGB")
        self.scale      = 1.0
        self._rect_id   = None
        self._start     = None
        self._end       = None

        root.title("Template Crop Helper — drag to select, then save")
        root.configure(bg="#1e1e1e")
        self._build_ui()
        self._load_image()
        self._refresh_status()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        r = self.root

        # Top bar: file picker + status
        top = tk.Frame(r, bg="#1e1e1e")
        top.pack(fill="x", padx=8, pady=4)

        tk.Button(top, text="Open screenshot…", command=self._open_file,
                  bg="#444", fg="white", relief="flat", padx=8).pack(side="left")

        self._file_lbl = tk.Label(top, text="", bg="#1e1e1e", fg="#aaa",
                                  font=("Consolas", 9))
        self._file_lbl.pack(side="left", padx=8)

        # Canvas
        self.canvas = tk.Canvas(r, bg="#111", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Bottom bar: template buttons + save
        bot = tk.Frame(r, bg="#1e1e1e")
        bot.pack(fill="x", padx=8, pady=6)

        tk.Label(bot, text="Save crop as:", bg="#1e1e1e", fg="white",
                 font=("Helvetica", 10)).pack(side="left", padx=(0, 6))

        self._tpl_var = tk.StringVar(value=TEMPLATES[0][0])
        self._btn_refs = {}
        for name, _ in TEMPLATES:
            btn = tk.Button(
                bot, text=name,
                command=lambda n=name: self._tpl_var.set(n),
                relief="flat", padx=6, pady=3,
                font=("Consolas", 9),
            )
            btn.pack(side="left", padx=3)
            self._btn_refs[name] = btn

        self._save_btn = tk.Button(
            bot, text="Save crop", command=self._save,
            bg="#4CAF50", fg="white", relief="flat",
            padx=10, pady=3, font=("Helvetica", 10, "bold"),
        )
        self._save_btn.pack(side="right", padx=4)

        # Selection info label
        self._sel_lbl = tk.Label(r, text="No selection", bg="#1e1e1e", fg="#888",
                                 font=("Consolas", 9))
        self._sel_lbl.pack(pady=(0, 4))

        # Status panel — which templates exist
        self._status_frame = tk.Frame(r, bg="#1e1e1e")
        self._status_frame.pack(fill="x", padx=8, pady=(0, 6))
        self._status_labels = {}
        for name, desc in TEMPLATES:
            row = tk.Frame(self._status_frame, bg="#1e1e1e")
            row.pack(anchor="w")
            lbl = tk.Label(row, text="", bg="#1e1e1e",
                           font=("Consolas", 9), width=3)
            lbl.pack(side="left")
            tk.Label(row, text=f"{name:<28} {desc}",
                     bg="#1e1e1e", fg="#aaa",
                     font=("Consolas", 9)).pack(side="left")
            self._status_labels[name] = lbl

    # ── Image loading ─────────────────────────────────────────────────────────
    def _load_image(self):
        sw = self.root.winfo_screenwidth()  - 80
        sh = self.root.winfo_screenheight() - 260
        iw, ih = self.src_image.size
        self.scale = min(sw / iw, sh / ih, 1.0)
        nw, nh = round(iw * self.scale), round(ih * self.scale)

        display = self.src_image.resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(display)
        self.canvas.config(width=nw, height=nh)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open screenshot",
            filetypes=[("PNG images", "*.png *.PNG"), ("All files", "*.*")],
            initialdir=IMAGES_DIR,
        )
        if path:
            self.src_image = Image.open(path).convert("RGB")
            self._file_lbl.config(text=os.path.basename(path))
            self._load_image()
            self._start = self._end = None
            self._sel_lbl.config(text="No selection")

    # ── Rubber-band selection ─────────────────────────────────────────────────
    def _on_press(self, event):
        self._start = (event.x, event.y)
        self._end   = None
        if self._rect_id:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

    def _on_drag(self, event):
        if not self._start:
            return
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        x0, y0 = self._start
        self._rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="#00ff88", width=2,
        )

    def _on_release(self, event):
        if not self._start:
            return
        self._end = (event.x, event.y)
        x0, y0 = self._start
        x1, y1 = self._end
        px0 = round(min(x0, x1) / self.scale)
        py0 = round(min(y0, y1) / self.scale)
        px1 = round(max(x0, x1) / self.scale)
        py1 = round(max(y0, y1) / self.scale)
        w, h = px1 - px0, py1 - py0
        self._sel_lbl.config(
            text=f"Selection: ({px0}, {py0})  {w}×{h} px  "
                 f"→ will save as: {self._tpl_var.get()}",
            fg="#00ff88",
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save(self):
        if not self._start or not self._end:
            messagebox.showwarning("No selection", "Drag to select a region first.")
            return

        x0, y0 = self._start
        x1, y1 = self._end
        px0 = round(min(x0, x1) / self.scale)
        py0 = round(min(y0, y1) / self.scale)
        px1 = round(max(x0, x1) / self.scale)
        py1 = round(max(y0, y1) / self.scale)

        if px1 - px0 < 4 or py1 - py0 < 4:
            messagebox.showwarning("Too small", "Selection is too small — try again.")
            return

        crop = self.src_image.crop((px0, py0, px1, py1))
        name = self._tpl_var.get()
        dest = os.path.join(IMAGES_DIR, name)
        crop.save(dest)
        print(f"Saved: {dest}  ({crop.width}×{crop.height})")
        self._refresh_status()

        # Auto-advance to next missing template
        missing = [n for n, _ in TEMPLATES
                   if not os.path.isfile(os.path.join(IMAGES_DIR, n))]
        if missing:
            self._tpl_var.set(missing[0])
        else:
            messagebox.showinfo("All done!",
                                "Both templates saved.\nYou can close this window.")

    # ── Status panel ──────────────────────────────────────────────────────────
    def _refresh_status(self):
        for name, lbl in self._status_labels.items():
            exists = os.path.isfile(os.path.join(IMAGES_DIR, name))
            lbl.config(text=" ✓" if exists else " ○",
                       fg="#4CAF50" if exists else "#888")
            btn = self._btn_refs[name]
            btn.config(
                bg="#2d6a2d" if exists else "#555",
                fg="white",
                relief="sunken" if self._tpl_var.get() == name else "flat",
            )
        # Highlight selected button
        sel = self._tpl_var.get()
        for name, btn in self._btn_refs.items():
            btn.config(relief="sunken" if name == sel else "flat")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    default_shot = os.path.join(IMAGES_DIR, "debug_window.PNG")
    shot_path    = sys.argv[1] if len(sys.argv) > 1 else default_shot

    if not os.path.isfile(shot_path):
        print(f"Screenshot not found: {shot_path}")
        print("Run a Team Scan first — it saves images/debug_window.PNG automatically.")
        sys.exit(1)

    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{min(1400, sw - 40)}x{min(900, sh - 40)}+20+20")

    app = CropSelector(root, shot_path)
    app._file_lbl.config(text=os.path.basename(shot_path))
    root.mainloop()
