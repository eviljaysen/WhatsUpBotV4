"""bot/ocr_model.py — Character-level CNN for game OCR.

A lightweight image classifier trained on individual character crops extracted
from labeled OCR training samples. The game uses a consistent font, so a small
CNN (~20K params) achieves near-perfect accuracy once trained.

Pipeline: conv image → segment characters → classify each → join → return text

Training data comes from training_data/ocr/{field}/{ts}_{value}_conv.png files
that are auto-collected during scans.

Exports:
    segment_characters(img)           — vertical projection segmentation
    predict_text(img, field="")       — segment + classify + join
    train_model(epochs=0)             — train CNN on all OCR fields
    get_training_stats()              — char counts + sample counts
    needs_training()                  — True if new samples since last train
    is_model_ready()                  — True if trained model exists
"""

import os
import json
import threading
import numpy as np
from PIL import Image as Img
from collections import Counter

from bot.config import BASE_DIR, get_logger

_log = get_logger("ocr_model")

# ── Paths ─────────────────────────────────────────────────────────────────────
OCR_TRAIN_DIR     = os.path.join(BASE_DIR, "training_data", "ocr")
MODEL_PATH        = os.path.join(BASE_DIR, "training_data", "ocr_char_model.pth")
CLASSES_PATH      = os.path.join(BASE_DIR, "training_data", "ocr_char_classes.json")
_TRAIN_COUNT_PATH = os.path.join(BASE_DIR, "training_data", "ocr_train_count.json")

# ── Character image dimensions ────────────────────────────────────────────────
CHAR_W, CHAR_H = 20, 32

# ── Lazy-loaded model state ───────────────────────────────────────────────────
_model = None
_classes = None
_device = None
_model_lock = threading.Lock()

# Load persisted train count — survives app restarts so needs_training() stays accurate
def _load_train_count() -> int:
    try:
        with open(_TRAIN_COUNT_PATH, "r") as f:
            return int(json.load(f).get("count", 0))
    except Exception:
        return 0

_last_train_count = _load_train_count()


# ── Character segmentation ────────────────────────────────────────────────────
def _detect_inverted(arr: np.ndarray) -> bool:
    """Check if image is white-on-black (inverted) by sampling the border."""
    h, w = arr.shape
    border = np.concatenate([
        arr[0, :], arr[-1, :],      # top and bottom rows
        arr[:, 0], arr[:, -1],      # left and right columns
    ])
    return np.mean(border) < 128


def segment_characters(img) -> list:
    """Segment a conv image into individual character crops.

    Uses vertical projection profiling: count dark pixels per column,
    find gaps between ink runs, extract bounding boxes.

    Args:
        img: PIL Image (B&W conv image, any size)

    Returns:
        List of numpy arrays (CHAR_H × CHAR_W, float32 0..1), left-to-right.
        Empty list if segmentation fails.
    """
    arr = np.asarray(img.convert("L"), dtype=np.uint8)

    # Auto-invert white-on-black images (team_points, opp_points)
    if _detect_inverted(arr):
        arr = 255 - arr

    # Binarize: ink=1, background=0
    ink = (arr < 128).astype(np.uint8)
    h, w = ink.shape

    # Remove horizontal bars (underlines, borders): rows where >50% is ink
    # but only if they're short (< 20% of image height)
    row_ink = ink.sum(axis=1)
    row_thresh = w * 0.4
    bar_rows = row_ink > row_thresh
    # Only blank bars that are thin (contiguous run < 20% of height)
    in_bar = False
    bar_start = 0
    for y in range(h):
        if bar_rows[y] and not in_bar:
            bar_start = y
            in_bar = True
        elif not bar_rows[y] and in_bar:
            bar_len = y - bar_start
            if bar_len < h * 0.2:
                ink[bar_start:y, :] = 0
            in_bar = False
    if in_bar:
        bar_len = h - bar_start
        if bar_len < h * 0.2:
            ink[bar_start:, :] = 0

    # Vertical projection: dark pixel count per column
    col_sum = ink.sum(axis=0)

    # Find ink runs (contiguous columns with ink > 0)
    runs = []
    in_run = False
    start = 0
    for x in range(w):
        if col_sum[x] > 0 and not in_run:
            start = x
            in_run = True
        elif col_sum[x] == 0 and in_run:
            runs.append((start, x))
            in_run = False
    if in_run:
        runs.append((start, w))

    if not runs:
        return []

    # Extract bounding boxes with vertical trim
    boxes = []
    for x0, x1 in runs:
        col_slice = ink[:, x0:x1]
        row_sum = col_slice.sum(axis=1)
        rows_with_ink = np.where(row_sum > 0)[0]
        if len(rows_with_ink) == 0:
            continue
        y0 = rows_with_ink[0]
        y1 = rows_with_ink[-1] + 1
        bw = x1 - x0
        bh = y1 - y0
        boxes.append((x0, y0, bw, bh))

    if not boxes:
        return []

    # Filter artifacts: compute outlier-aware median width.
    # If the widest run is much larger than the second widest (e.g. heart icon),
    # exclude it from the median calculation.
    widths = [b[2] for b in boxes]
    heights = [b[3] for b in boxes]
    sorted_w = sorted(widths)
    if (len(sorted_w) >= 3
            and sorted_w[-1] > sorted_w[-2] * 1.8):
        clean_w = sorted_w[:-1]
    else:
        clean_w = sorted_w
    med_w = clean_w[len(clean_w) // 2]
    med_h = sorted(heights)[len(heights) // 2]

    filtered = []
    for x0, y0, bw, bh in boxes:
        # Skip too wide (icons, badge borders): > 2× clean median width
        if med_w > 3 and bw > med_w * 2.0:
            continue
        # Skip too small (noise): < 3px wide or < 5px tall
        if bw < 3 or bh < 5:
            continue
        # Skip very wide-and-short (underlines): aspect ratio > 3
        if bh > 0 and bw / bh > 3.0:
            continue
        # Skip very tall-and-thin (badge vertical borders)
        if bw > 0 and bh / bw > 8.0 and bw < med_w * 0.3:
            continue
        filtered.append((x0, y0, bw, bh))

    if not filtered:
        return []

    # Split touching characters: if a box is much wider than median,
    # try two strategies in order:
    # 1. Adaptive gap: for very wide runs, re-segment using relaxed ink
    #    threshold (handles underline-connected chars in max_points).
    # 2. Valley-based: split at the deepest ink valley.
    min_w = min(b[2] for b in filtered) if filtered else 0
    final = []
    for x0, y0, bw, bh in filtered:
        col_slice = ink[y0:y0 + bh, x0:x0 + bw]
        profile = col_slice.sum(axis=0)
        max_ink = int(profile.max()) if len(profile) > 0 else 0

        # Strategy 1: Adaptive gap for very wide runs (> min_w × 2.5)
        if min_w >= 3 and bw > min_w * 2.5 and max_ink > 10:
            local_thresh = max(1, int(max_ink * 0.20))
            sub_runs = []
            in_sub = False
            sub_start = 0
            for dx in range(bw):
                if profile[dx] > local_thresh and not in_sub:
                    sub_start = dx
                    in_sub = True
                elif profile[dx] <= local_thresh and in_sub:
                    sub_runs.append((sub_start, dx))
                    in_sub = False
            if in_sub:
                sub_runs.append((sub_start, bw))
            if len(sub_runs) >= 2:
                for sx0, sx1 in sub_runs:
                    sw = sx1 - sx0
                    if sw >= 3:
                        final.append((x0 + sx0, y0, sw, bh))
                continue

        # Strategy 2: Valley-based split (original 1.6× threshold)
        if med_w > 5 and bw > med_w * 1.6:
            margin = max(3, bw // 5)
            mid_profile = profile[margin:bw - margin]
            if len(mid_profile) > 0:
                split_rel = margin + np.argmin(mid_profile)
                if max_ink > 0 and profile[split_rel] < max_ink * 0.4:
                    final.append((x0, y0, split_rel, bh))
                    final.append((x0 + split_rel, y0, bw - split_rel, bh))
                    continue
        final.append((x0, y0, bw, bh))

    # Sort left-to-right
    final.sort(key=lambda b: b[0])

    # Extract and normalize crops
    crops = []
    for x0, y0, bw, bh in final:
        crop = arr[y0:y0 + bh, x0:x0 + bw]
        # Resize to standard dimensions, preserving aspect ratio with padding
        pil_crop = Img.fromarray(crop).convert("L")
        # Fit into CHAR_W × CHAR_H box
        pil_crop.thumbnail((CHAR_W, CHAR_H), Img.LANCZOS)
        # Center on white canvas
        canvas = Img.new("L", (CHAR_W, CHAR_H), 255)
        ox = (CHAR_W - pil_crop.width) // 2
        oy = (CHAR_H - pil_crop.height) // 2
        canvas.paste(pil_crop, (ox, oy))
        crops.append(np.asarray(canvas, dtype=np.float32) / 255.0)

    return crops


# ── Label extraction ──────────────────────────────────────────────────────────
def _label_from_filename(field: str, fname: str) -> str:
    """Extract the character sequence from a training sample filename.

    Timer: value 1653 → "16h53m" (hours * 100 + minutes format)
    Bonus fields: value 85 → "85" (ignore the '+' in the image)
    Name: value is the player name string, e.g. "JAYSEN"
    All others: value as digit string, e.g. "1393007"
    """
    parts = fname.split("_")
    if len(parts) < 3:
        return ""
    value_str = parts[1]
    if field == "timer":
        try:
            v = int(value_str)
            hours = v // 100
            mins = v % 100
            return f"{hours}h{mins}m"
        except ValueError:
            return ""
    if field == "name":
        # Player names — rejoin all parts between timestamp and suffix
        # e.g. "123_NOT HEY_conv.png" → "NOT HEY"
        return "_".join(parts[1:-1]).replace("_conv.png", "")
    return value_str


# ── Dataset building ──────────────────────────────────────────────────────────
def _augment_char(arr: np.ndarray, rng: np.random.Generator) -> list:
    """Generate augmented variants of a character crop (H, W) float32 [0..1]."""
    variants = []
    for _ in range(3):  # 3 augmented copies → 4× total data
        aug = arr.copy()
        # Random shift ±1px x and y
        dx = rng.integers(-1, 2)
        dy = rng.integers(-1, 2)
        if dx != 0:
            aug = np.roll(aug, dx, axis=1)
            if dx > 0:
                aug[:, :dx] = rng.uniform(0.9, 1.0)
            else:
                aug[:, dx:] = rng.uniform(0.9, 1.0)
        if dy != 0:
            aug = np.roll(aug, dy, axis=0)
            if dy > 0:
                aug[:dy, :] = rng.uniform(0.9, 1.0)
            else:
                aug[dy:, :] = rng.uniform(0.9, 1.0)
        # Brightness jitter ±10%
        aug = np.clip(aug * rng.uniform(0.9, 1.1), 0.0, 1.0)
        # Gaussian noise
        aug = np.clip(aug + rng.normal(0, 0.02, aug.shape).astype(np.float32),
                      0.0, 1.0)
        variants.append(aug)
    return variants


def _build_char_dataset(augment: bool = True):
    """Build character dataset from all OCR training samples.

    Segments each conv image, aligns with label string, and collects
    (crop, char_class) pairs. Discards samples where segment count
    doesn't match label length.

    Returns:
        (images, labels, class_names) or (None, None, [])
    """
    if not os.path.isdir(OCR_TRAIN_DIR):
        return None, None, []

    # Collect all character samples
    char_samples = []  # [(crop_array, char_str)]

    for field in sorted(os.listdir(OCR_TRAIN_DIR)):
        field_dir = os.path.join(OCR_TRAIN_DIR, field)
        if not os.path.isdir(field_dir):
            continue

        for fname in os.listdir(field_dir):
            if not fname.endswith("_conv.png"):
                continue

            label = _label_from_filename(field, fname)
            if not label:
                continue

            path = os.path.join(field_dir, fname)
            try:
                img = Img.open(path).convert("L")
                crops = segment_characters(img)
            except Exception as e:
                _log.warning("Failed to segment %s: %s", path, e)
                continue

            # Align: segment count must match label length
            if len(crops) != len(label):
                # Tolerate off-by-one from '+' prefix in bonus fields
                if field in ("team_bonus", "opp_bonus") and len(crops) == len(label) + 1:
                    crops = crops[1:]  # skip the '+' crop
                elif field in ("team_bonus", "opp_bonus") and len(crops) == len(label) - 1:
                    continue  # '+' merged with first digit, skip
                else:
                    continue  # mismatch — skip sample

            for crop, char in zip(crops, label):
                char_samples.append((crop, char))

    if not char_samples:
        return None, None, []

    # Build class list from observed characters
    all_chars = sorted(set(s[1] for s in char_samples))
    char_to_idx = {c: i for i, c in enumerate(all_chars)}

    rng = np.random.default_rng(42)
    images = []
    labels = []

    for crop, char in char_samples:
        images.append(crop[np.newaxis, :, :])  # add channel dim
        labels.append(char_to_idx[char])
        if augment:
            for aug in _augment_char(crop, rng):
                images.append(aug[np.newaxis, :, :])
                labels.append(char_to_idx[char])

    return (np.array(images, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            all_chars)


# ── CNN model ────────────────────────────────────────────────────────────────
def _build_model(num_classes: int):
    """Build a small CNN for character classification.

    Architecture: 2 conv blocks → global avg pool → FC
    Input: 1×32×20 grayscale. ~20K params.
    """
    import torch.nn as nn

    class CharCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: 1×32×20 → 32×16×10
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # Block 2: 32×16×10 → 64×8×5
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return CharCNN(num_classes)


# ── Training stats ────────────────────────────────────────────────────────────
def get_training_stats() -> dict:
    """Return training data statistics.

    Returns dict with:
        total_samples: number of conv images across all fields
        fields: {field_name: sample_count}
        total_chars: estimated character count (from segmentation)
    """
    stats = {"total_samples": 0, "fields": {}}
    if not os.path.isdir(OCR_TRAIN_DIR):
        return stats
    for field in sorted(os.listdir(OCR_TRAIN_DIR)):
        fdir = os.path.join(OCR_TRAIN_DIR, field)
        if os.path.isdir(fdir):
            count = len([f for f in os.listdir(fdir) if f.endswith("_conv.png")])
            if count > 0:
                stats["fields"][field] = count
                stats["total_samples"] += count
    return stats


def needs_training() -> bool:
    """Check if new OCR samples have been collected since last training."""
    stats = get_training_stats()
    total = stats["total_samples"]
    return total != _last_train_count and total > 0


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(epochs: int = 0, lr: float = 0.001) -> dict:
    """Train the character CNN on all OCR training samples.

    Args:
        epochs: training epochs (0 = auto-scale)
        lr: learning rate

    Returns:
        dict with training results or {"error": "reason"}
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    global _last_train_count

    images, labels, class_names = _build_char_dataset(augment=True)
    if images is None or len(class_names) < 2:
        _log.warning("Not enough data to train")
        return {"error": "insufficient data"}

    num_classes = len(class_names)
    counts = Counter(labels.tolist())

    # Auto-scale epochs
    if epochs <= 0:
        n = len(images)
        if n < 200:
            epochs = 15
        elif n < 1000:
            epochs = 25
        elif n < 5000:
            epochs = 35
        else:
            epochs = 45

    _log.info("Training: %d classes, %d samples, %d epochs",
              num_classes, len(images), epochs)
    for i, name in enumerate(class_names):
        _log.debug("  '%s': %d samples", name, counts.get(i, 0))

    device = torch.device("cpu")
    X = torch.from_numpy(images)
    y = torch.from_numpy(labels)

    # 80/20 split
    n = len(X)
    perm = torch.randperm(n)
    split = max(1, int(n * 0.8))
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = _build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    best_acc = 0.0
    best_loss = float("inf")
    patience_counter = 0
    EARLY_STOP_PATIENCE = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X[val_idx].to(device))
                val_acc = float(
                    (val_out.argmax(1) == y[val_idx].to(device)).sum()
                ) / len(val_idx)
                best_acc = max(best_acc, val_acc)
        else:
            val_acc = -1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _log.debug("  Epoch %d/%d: loss=%.4f val_acc=%.3f",
                       epoch + 1, epochs, avg_loss, val_acc)

        # Early stopping
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                _log.debug("  Early stop at epoch %d (loss plateau)", epoch + 1)
                break

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "class_names": class_names,
    }, MODEL_PATH)

    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)

    # Invalidate cached model + update train count
    with _model_lock:
        global _model, _classes
        _model = None
        _classes = None
        stats = get_training_stats()
        _last_train_count = stats["total_samples"]
    # Persist count to disk so needs_training() stays accurate across app restarts
    try:
        with open(_TRAIN_COUNT_PATH, "w") as f:
            json.dump({"count": _last_train_count}, f)
    except Exception as e:
        _log.warning("Could not persist train count: %s", e)

    # Final accuracy
    model.eval()
    with torch.no_grad():
        all_pred = model(X.to(device)).argmax(1)
        train_acc = float((all_pred == y.to(device)).sum()) / len(y)

    result = {
        "accuracy": train_acc,
        "val_accuracy": best_acc,
        "num_classes": num_classes,
        "num_samples": len(images),
        "class_names": class_names,
    }
    _log.info("Training complete: acc=%.3f val_acc=%.3f", train_acc, best_acc)
    return result


# ── Prediction ────────────────────────────────────────────────────────────────
def _load_model():
    """Load the trained model (lazy, thread-safe)."""
    global _model, _classes, _device

    with _model_lock:
        if _model is not None:
            return _model, _classes

        if not os.path.isfile(MODEL_PATH):
            return None, None

        try:
            import torch
            _device = torch.device("cpu")
            ckpt = torch.load(MODEL_PATH, map_location=_device,
                              weights_only=False)
            _classes = ckpt["class_names"]
            _model = _build_model(ckpt["num_classes"]).to(_device)
            _model.load_state_dict(ckpt["state_dict"])
            _model.eval()
            _log.info("Loaded model: %d classes", len(_classes))
            return _model, _classes
        except Exception as e:
            _log.error("Failed to load model: %s", e, exc_info=True)
            return None, None


_DIGIT_FIELDS = {"team_points", "opp_points", "max_points", "slot_hp", "slot_atk"}
_TIMER_FIELDS = {"timer"}


def predict_text(img, field: str = "") -> tuple:
    """Segment and classify characters from a conv image.

    Args:
        img: PIL Image (B&W conv image)
        field: optional field name — used to restrict output classes
               (digit-only fields won't predict letters)

    Returns:
        (text: str, avg_confidence: float). ('', 0.0) if unavailable.
    """
    model, classes = _load_model()
    if model is None:
        return '', 0.0

    crops = segment_characters(img)
    if not crops:
        return '', 0.0

    # Build allowed class mask based on field type.
    # Digit-only fields (scores, HP/ATK) restrict to 0-9.
    # Timer fields restrict to 0-9 + h + m.
    # All other fields (names, unknown) allow all classes.
    allowed_mask = None
    if field in _DIGIT_FIELDS:
        allowed_mask = [i for i, c in enumerate(classes) if c.isdigit()]
    elif field in _TIMER_FIELDS:
        allowed_mask = [i for i, c in enumerate(classes)
                        if c.isdigit() or c in ('h', 'm')]

    try:
        import torch

        batch = torch.stack([
            torch.from_numpy(c).unsqueeze(0) for c in crops
        ]).to(_device)

        with torch.no_grad():
            logits = model(batch)
            # Mask out disallowed classes before softmax
            if allowed_mask is not None:
                mask = torch.full(logits.shape, float('-inf'), device=_device)
                for idx in allowed_mask:
                    mask[:, idx] = 0.0
                logits = logits + mask
            probs = torch.softmax(logits, dim=1)
            confs, idxs = probs.max(dim=1)

        chars = []
        total_conf = 0.0
        for i in range(len(crops)):
            conf = confs[i].item()
            if conf >= 0.3:  # skip very low confidence
                chars.append(classes[idxs[i].item()])
                total_conf += conf

        text = ''.join(chars)
        avg_conf = total_conf / len(chars) if chars else 0.0
        return text, avg_conf
    except Exception as e:
        _log.error("Prediction error: %s", e, exc_info=True)
        return '', 0.0


def is_model_ready() -> bool:
    """Check if a trained model file exists."""
    return os.path.isfile(MODEL_PATH)
