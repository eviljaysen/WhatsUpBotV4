"""tools/collect_training_data.py — Collect OCR training samples from game screenshots.

Builds a labeled dataset from debug images for fine-tuning EasyOCR or training
a custom OCR model. Also supports live capture from the game window.

Usage:
    python tools/collect_training_data.py --mode timer      # label timer debug images
    python tools/collect_training_data.py --mode name       # label name debug images
    python tools/collect_training_data.py --mode badge      # label badge debug images
    python tools/collect_training_data.py --mode live       # capture live from game
    python tools/collect_training_data.py --export          # export to EasyOCR format
    python tools/collect_training_data.py --stats           # show dataset statistics

Workflow:
    1. Run scans to generate debug_*_raw.PNG / debug_*_conv.PNG files in images/
    2. Run this script with --mode to label each image interactively
    3. Labels are saved to training_data/labels.json
    4. Use --export to generate EasyOCR fine-tuning directory structure
"""

import argparse
import json
import os
import re
import shutil
import sys
import tkinter as tk
from tkinter import ttk
from datetime import datetime

from PIL import Image as Img, ImageTk

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.config import IMAGES_DIR, BASE_DIR

TRAINING_DIR = os.path.join(BASE_DIR, "training_data")
LABELS_PATH = os.path.join(TRAINING_DIR, "labels.json")
SAMPLES_DIR = os.path.join(TRAINING_DIR, "samples")


def _ensure_dirs():
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)


def _load_labels() -> list:
    if os.path.isfile(LABELS_PATH):
        with open(LABELS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_labels(labels: list):
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


def _find_debug_images(mode: str) -> list:
    """Find debug images in images/ matching the given mode."""
    patterns = {
        "timer": ["debug_timer_raw.PNG", "debug_timer_conv.PNG"],
        "name": ["debug_name_raw.PNG", "debug_name_conv.PNG", "debug_name_cjk.PNG"],
        "badge": [
            "debug_team_bonus_raw.PNG", "debug_team_bonus_conv.PNG",
            "debug_opp_bonus_raw.PNG", "debug_opp_bonus_conv.PNG",
        ],
        "score": [
            "debug_team_points_raw.PNG", "debug_team_points_conv.PNG",
            "debug_opp_points_raw.PNG", "debug_opp_points_conv.PNG",
            "debug_max_points_raw.PNG", "debug_max_points_conv.PNG",
        ],
    }
    target_files = patterns.get(mode, [])
    found = []
    for fname in target_files:
        path = os.path.join(IMAGES_DIR, fname)
        if os.path.isfile(path):
            found.append(path)
    return found


class LabelingApp:
    """Interactive tkinter app for labeling OCR training images."""

    def __init__(self, images: list, mode: str):
        self.images = images
        self.mode = mode
        self.index = 0
        self.labels = _load_labels()
        self.labeled_paths = {l["image"] for l in self.labels}

        self.root = tk.Tk()
        self.root.title(f"OCR Training — {mode}")
        self.root.geometry("800x500")

        # Image display
        self.canvas = tk.Canvas(self.root, bg="gray20", height=300)
        self.canvas.pack(fill=tk.X, padx=10, pady=10)
        self._photo = None  # prevent GC

        # Info label
        self.info_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.info_var,
                  font=("Consolas", 11)).pack(pady=5)

        # Input frame
        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=10)
        ttk.Label(input_frame, text="Ground truth:").pack(side=tk.LEFT, padx=5)
        self.entry = ttk.Entry(input_frame, width=40, font=("Consolas", 14))
        self.entry.pack(side=tk.LEFT, padx=5)
        self.entry.bind("<Return>", self._on_submit)

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Save (Enter)", command=self._on_submit).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Skip", command=self._on_skip).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Quit", command=self.root.destroy).pack(
            side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var,
                  foreground="green").pack(pady=5)

        self._show_current()

    def _show_current(self):
        if self.index >= len(self.images):
            self.info_var.set("All images labeled!")
            self.entry.config(state="disabled")
            return

        path = self.images[self.index]
        fname = os.path.basename(path)
        already = " (already labeled)" if path in self.labeled_paths else ""
        self.info_var.set(
            f"[{self.index + 1}/{len(self.images)}] {fname}{already}")

        # Show image
        try:
            img = Img.open(path)
            # Scale up small images for visibility
            scale = max(1, min(4, 280 // max(img.height, 1)))
            if scale > 1:
                img = img.resize((img.width * scale, img.height * scale),
                                 Img.NEAREST)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(400, 150, image=self._photo)
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(400, 150, text=f"Error: {e}",
                                    fill="red", font=("Consolas", 12))

        self.entry.delete(0, tk.END)
        self.entry.focus()

    def _on_submit(self, event=None):
        text = self.entry.get().strip()
        if not text or self.index >= len(self.images):
            return

        path = self.images[self.index]
        # Copy image to samples dir with timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.basename(path)
        sample_name = f"{self.mode}_{ts}_{fname}"
        sample_path = os.path.join(SAMPLES_DIR, sample_name)
        shutil.copy2(path, sample_path)

        # Remove old label for this source if exists
        self.labels = [l for l in self.labels if l.get("source") != path]
        self.labels.append({
            "image": sample_path,
            "source": path,
            "text": text,
            "type": self.mode,
            "timestamp": ts,
        })
        _save_labels(self.labels)
        self.labeled_paths.add(path)
        self.status_var.set(f"Saved: '{text}' → {sample_name}")

        self.index += 1
        self._show_current()

    def _on_skip(self):
        self.index += 1
        self._show_current()

    def run(self):
        self.root.mainloop()


def _live_capture(mode: str):
    """Capture live screenshots from the game for labeling."""
    try:
        from bot.window import detect_window
        from bot.templates import _HUD
        import pyautogui
    except ImportError as e:
        print(f"Cannot import bot modules: {e}")
        return

    region_map = {
        "timer": "timer",
        "name": "enemy_name",
        "badge": "opp_bonus",
        "score": "team_points",
    }
    hud_key = region_map.get(mode)
    if not hud_key:
        print(f"No HUD region defined for mode '{mode}'")
        return

    win = detect_window()
    region = win.hud(hud_key)
    _ensure_dirs()

    print(f"Live capture mode: '{mode}' from HUD region '{hud_key}'")
    print(f"Region: {region}")
    print("Press Enter to capture, 'q' to quit")

    captures = []
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        shot = pyautogui.screenshot(region=region)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"live_{mode}_{ts}.PNG"
        path = os.path.join(SAMPLES_DIR, fname)
        shot.save(path)
        captures.append(path)
        print(f"  Captured: {fname} ({shot.width}x{shot.height})")

    if captures:
        print(f"\nCaptured {len(captures)} images. Run with --mode {mode} to label them.")


def _export_easyocr():
    """Export labeled data to EasyOCR fine-tuning format."""
    labels = _load_labels()
    if not labels:
        print("No labels found. Run labeling first.")
        return

    export_dir = os.path.join(TRAINING_DIR, "easyocr_export")

    for label_type in set(l["type"] for l in labels):
        type_dir = os.path.join(export_dir, label_type)
        os.makedirs(type_dir, exist_ok=True)

        type_labels = [l for l in labels if l["type"] == label_type]
        csv_lines = []
        for i, l in enumerate(type_labels):
            src = l["image"]
            if not os.path.isfile(src):
                print(f"  Missing: {src}")
                continue
            ext = os.path.splitext(src)[1]
            dst_name = f"{i:04d}{ext}"
            shutil.copy2(src, os.path.join(type_dir, dst_name))
            # EasyOCR format: filename\ttext
            csv_lines.append(f"{dst_name}\t{l['text']}")

        csv_path = os.path.join(type_dir, "labels.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(csv_lines))

        print(f"  {label_type}: {len(csv_lines)} samples → {type_dir}")

    print(f"\nExported to {export_dir}")
    print("Use these directories with EasyOCR's trainer module for fine-tuning.")


def _train_name_model(epochs: int = 30):
    """Train the CNN name classifier on auto-collected samples."""
    from bot.name_model import train_model, get_training_stats

    stats = get_training_stats()
    if not stats:
        print("No name training samples found.")
        print("Run a few scans first — samples are auto-collected on every successful name match.")
        return

    print(f"\nName Model Training Data")
    print(f"{'-' * 40}")
    total = sum(stats.values())
    print(f"Total samples: {total}")
    print(f"Players: {len(stats)}")
    for name, count in sorted(stats.items()):
        bar = '#' * min(count, 40)
        print(f"  {name:16s} {count:3d}  {bar}")
    print()

    if total < 20:
        print(f"Recommended: ≥10 samples per player for good accuracy.")
        print(f"Run more scans to collect samples automatically.")
        resp = input("Train anyway? (y/N): ").strip().lower()
        if resp != 'y':
            return

    result = train_model(epochs=epochs)
    if "error" in result:
        print(f"Training failed: {result['error']}")
    else:
        print(f"\nResults:")
        print(f"  Accuracy:       {result['accuracy']:.1%}")
        print(f"  Val Accuracy:   {result['val_accuracy']:.1%}")
        print(f"  Classes:        {result['num_classes']}")
        print(f"  Samples:        {result['num_samples']}")


def _show_stats():
    """Show dataset statistics."""
    # Name model stats
    from bot.name_model import get_training_stats, is_model_ready
    name_stats = get_training_stats()
    if name_stats:
        print(f"\nName Model Training Data")
        print(f"{'-' * 40}")
        total = sum(name_stats.values())
        print(f"Total samples: {total}, Players: {len(name_stats)}")
        print(f"Model trained: {'Yes' if is_model_ready() else 'No'}")
        for name, count in sorted(name_stats.items()):
            bar = '#' * min(count, 40)
            print(f"  {name:16s} {count:3d}  {bar}")
        print()

    # General OCR label stats
    labels = _load_labels()
    if labels:
        by_type = {}
        for l in labels:
            t = l["type"]
            by_type.setdefault(t, []).append(l)

        print(f"OCR Training Labels")
        print(f"{'-' * 40}")
        print(f"Total samples: {len(labels)}")
        print()
        for t, items in sorted(by_type.items()):
            texts = [i["text"] for i in items]
            unique = len(set(texts))
            print(f"  {t:10s}: {len(items):4d} samples, {unique:3d} unique labels")
            from collections import Counter
            common = Counter(texts).most_common(5)
            for text, count in common:
                print(f"    {text!r:20s} × {count}")
        print()

    if not name_stats and not labels:
        print("No training data yet. Run scans to auto-collect name samples.")


def main():
    parser = argparse.ArgumentParser(
        description="Collect and label OCR training data from game screenshots")
    parser.add_argument("--mode", choices=["timer", "name", "badge", "score", "live"],
                        help="Type of images to label")
    parser.add_argument("--export", action="store_true",
                        help="Export labeled data to EasyOCR format")
    parser.add_argument("--train", action="store_true",
                        help="Train the CNN name classifier on auto-collected samples")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--stats", action="store_true",
                        help="Show dataset statistics")
    args = parser.parse_args()

    _ensure_dirs()

    if args.stats:
        _show_stats()
        return

    if args.train:
        _train_name_model(epochs=args.epochs)
        return

    if args.export:
        _export_easyocr()
        return

    if not args.mode:
        parser.print_help()
        return

    if args.mode == "live":
        mode = input("Capture type (timer/name/badge/score): ").strip()
        _live_capture(mode)
        return

    # Find debug images + any previously captured live images
    images = _find_debug_images(args.mode)
    # Also include live captures for this mode
    if os.path.isdir(SAMPLES_DIR):
        for f in sorted(os.listdir(SAMPLES_DIR)):
            if f.startswith(f"live_{args.mode}_") and f.endswith(".PNG"):
                images.append(os.path.join(SAMPLES_DIR, f))

    if not images:
        print(f"No images found for mode '{args.mode}'.")
        print(f"Run a scan first to generate debug images in {IMAGES_DIR}")
        print(f"Or use --mode live to capture from the game window.")
        return

    print(f"Found {len(images)} images for '{args.mode}'")
    app = LabelingApp(images, args.mode)
    app.run()

    _show_stats()


if __name__ == "__main__":
    main()
