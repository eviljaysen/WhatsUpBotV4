"""bot/name_model.py — CNN classifier for player name recognition.

A lightweight image classifier trained on name-region screenshots. Each player
is a class (~25 classes). Much faster and more accurate than OCR once trained.

Self-improving: every successful OCR match auto-saves a training sample.
After collecting enough samples (10+ per player), train the model and it
becomes the first check in the name pipeline — before template matching.

Exports:
    save_training_sample(player, shot)  — save labeled name crop during scans
    predict_name(shot) -> (player, confidence) — classify a name screenshot
    train_model(epochs=30) -> dict     — train CNN on collected samples
    get_training_stats() -> dict       — sample counts per player
    is_model_ready() -> bool           — True if trained model exists
"""

import os
import json
import time
import threading
import numpy as np
from PIL import Image as Img
from collections import Counter

from bot.config import BASE_DIR, CFG

# ── Paths ─────────────────────────────────────────────────────────────────────
NAME_TRAIN_DIR = os.path.join(BASE_DIR, "training_data", "names")
MODEL_PATH = os.path.join(BASE_DIR, "training_data", "name_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "training_data", "name_classes.json")

# ── Image dimensions for the CNN ──────────────────────────────────────────────
IMG_W, IMG_H = 160, 40  # same as template thumbnails

# ── Lazy-loaded model state ───────────────────────────────────────────────────
_model = None
_classes = None
_device = None
_model_lock = threading.Lock()  # protects _model/_classes between scan + train threads
_last_train_count = 0   # total RAW samples at last training — skip if unchanged


def _ensure_dirs():
    os.makedirs(NAME_TRAIN_DIR, exist_ok=True)


# ── Training sample collection ────────────────────────────────────────────────
def save_training_sample(player: str, shot):
    """Save a name-region screenshot as a training sample for a player.

    Called automatically after every successful name match during scans.
    Caps at 50 samples per player to prevent disk bloat.

    Args:
        player: canonical player name (directory name)
        shot: PIL Image of the raw name region
    """
    if not player or not shot:
        return

    _ensure_dirs()
    player_dir = os.path.join(NAME_TRAIN_DIR, player)
    os.makedirs(player_dir, exist_ok=True)

    existing = [f for f in os.listdir(player_dir) if f.endswith(".png")]
    if len(existing) >= 50:
        # Remove oldest to make room
        oldest = sorted(existing)[0]
        os.remove(os.path.join(player_dir, oldest))

    ts = int(time.time() * 1000)
    path = os.path.join(player_dir, f"{ts}.png")
    # Save as grayscale thumbnail matching CNN input size
    gray = shot.convert("L").resize((IMG_W, IMG_H), Img.LANCZOS)
    gray.save(path)


def get_training_stats() -> dict:
    """Return sample counts per player. {player: count}"""
    _ensure_dirs()
    stats = {}
    if not os.path.isdir(NAME_TRAIN_DIR):
        return stats
    for d in sorted(os.listdir(NAME_TRAIN_DIR)):
        dpath = os.path.join(NAME_TRAIN_DIR, d)
        if os.path.isdir(dpath):
            count = len([f for f in os.listdir(dpath) if f.endswith(".png")])
            if count > 0:
                stats[d] = count
    return stats


def needs_training() -> bool:
    """Check if new samples have been collected since last training.

    Returns True only when the total sample count has changed,
    preventing expensive retraining after every scan when no new
    data was collected.
    """
    stats = get_training_stats()
    total = sum(stats.values())
    return total != _last_train_count and total > 0


# ── CNN Model Definition ─────────────────────────────────────────────────────
def _build_model(num_classes: int):
    """Build a small CNN for name classification.

    Architecture: 3 conv blocks → global avg pool → FC → softmax
    Input: 1×40×160 grayscale image
    ~50K params — trains in seconds on CPU
    """
    import torch
    import torch.nn as nn

    class NameCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: 1×40×160 → 32×20×80
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 2: 32×20×80 → 64×10×40
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 3: 64×10×40 → 128×5×20
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return NameCNN(num_classes)


# ── Data Augmentation ─────────────────────────────────────────────────────────
def _augment(arr: np.ndarray, rng: np.random.Generator) -> list:
    """Generate augmented variants of a single image array (H, W) float32 [0..1].

    Returns list of augmented arrays (does NOT include the original).
    Each variant applies random horizontal shift, brightness jitter, and noise.
    """
    variants = []
    for _ in range(4):  # 4 augmented copies per original → 5× total data
        aug = arr.copy()
        # Random horizontal shift ±8px (names can be slightly offset)
        dx = rng.integers(-8, 9)
        if dx != 0:
            aug = np.roll(aug, dx, axis=1)
            if dx > 0:
                aug[:, :dx] = rng.uniform(0.85, 1.0)  # fill with near-white
            else:
                aug[:, dx:] = rng.uniform(0.85, 1.0)
        # Random vertical shift ±2px
        dy = rng.integers(-2, 3)
        if dy != 0:
            aug = np.roll(aug, dy, axis=0)
            if dy > 0:
                aug[:dy, :] = rng.uniform(0.85, 1.0)
            else:
                aug[dy:, :] = rng.uniform(0.85, 1.0)
        # Brightness jitter ±15%
        factor = rng.uniform(0.85, 1.15)
        aug = np.clip(aug * factor, 0.0, 1.0)
        # Gaussian noise
        noise = rng.normal(0, 0.02, aug.shape).astype(np.float32)
        aug = np.clip(aug + noise, 0.0, 1.0)
        variants.append(aug)
    return variants


# ── Dataset Loading ───────────────────────────────────────────────────────────
def _load_dataset(augment: bool = True):
    """Load all training samples as (images, labels, class_names).

    Args:
        augment: if True, generate 4 augmented variants per sample (5× data)

    Returns:
        images: np.ndarray of shape (N, 1, IMG_H, IMG_W) float32 [0..1]
        labels: np.ndarray of shape (N,) int64
        class_names: list of player names (index = class label)
    """
    stats = get_training_stats()
    if not stats:
        return None, None, []

    class_names = sorted(stats.keys())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    rng = np.random.default_rng(42)
    images = []
    labels = []

    for player, count in stats.items():
        player_dir = os.path.join(NAME_TRAIN_DIR, player)
        for fname in os.listdir(player_dir):
            if not fname.endswith(".png"):
                continue
            path = os.path.join(player_dir, fname)
            try:
                img = Img.open(path).convert("L").resize((IMG_W, IMG_H), Img.LANCZOS)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                images.append(arr[np.newaxis, :, :])  # add channel dim
                labels.append(class_to_idx[player])
                # Augmented variants
                if augment:
                    for aug_arr in _augment(arr, rng):
                        images.append(aug_arr[np.newaxis, :, :])
                        labels.append(class_to_idx[player])
            except Exception as e:
                print(f"[name_model] Failed to load {path}: {e}")

    if not images:
        return None, None, []

    return (np.array(images, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            class_names)


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(epochs: int = 0, lr: float = 0.001,
                min_samples: int = 3) -> dict:
    """Train the CNN on collected samples.

    Args:
        epochs: training epochs (0 = auto-scale based on dataset size)
        lr: learning rate
        min_samples: minimum samples per player to include in training

    Returns:
        dict with training results: {accuracy, num_classes, num_samples, epoch_losses}
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    images, labels, class_names = _load_dataset(augment=True)
    if images is None or len(class_names) < 2:
        print("[name_model] Not enough data to train (need ≥2 players with samples)")
        return {"error": "insufficient data"}

    # Filter classes with too few samples
    counts = Counter(labels.tolist())
    valid_classes = {c for c, n in counts.items() if n >= min_samples}
    if len(valid_classes) < 2:
        print(f"[name_model] Need ≥{min_samples} samples per player. "
              f"Current: {dict(zip(class_names, [counts.get(i, 0) for i, _ in enumerate(class_names)]))}")
        return {"error": "insufficient samples per class"}

    # Remap to contiguous labels
    mask = np.array([l in valid_classes for l in labels])
    images = images[mask]
    labels = labels[mask]
    old_to_new = {old: new for new, old in enumerate(sorted(valid_classes))}
    labels = np.array([old_to_new[l] for l in labels], dtype=np.int64)
    class_names = [class_names[old] for old in sorted(valid_classes)]
    num_classes = len(class_names)

    # Auto-scale epochs based on dataset size (with augmentation, ~500 per player)
    if epochs <= 0:
        n_samples = len(images)
        if n_samples < 100:
            epochs = 20
        elif n_samples < 500:
            epochs = 30
        elif n_samples < 2000:
            epochs = 40
        else:
            epochs = 50

    print(f"[name_model] Training: {num_classes} players, {len(images)} samples, {epochs} epochs")
    for i, name in enumerate(class_names):
        n = int(np.sum(labels == i))
        print(f"  {name}: {n} samples")

    device = torch.device("cpu")
    X = torch.from_numpy(images)
    y = torch.from_numpy(labels)

    # 80/20 train/val split
    n = len(X)
    perm = torch.randperm(n)
    split = max(1, int(n * 0.8))
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx]) if len(val_idx) > 0 else None
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = _build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    epoch_losses = []
    best_acc = 0.0
    best_loss = float("inf")
    patience_counter = 0
    EARLY_STOP_PATIENCE = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)

        # Validation accuracy
        if val_ds and len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X[val_idx].to(device))
                val_pred = val_out.argmax(dim=1)
                val_acc = float((val_pred == y[val_idx].to(device)).sum()) / len(val_idx)
                if val_acc > best_acc:
                    best_acc = val_acc
        else:
            val_acc = -1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} val_acc={val_acc:.3f}")

        # Early stopping: stop when loss stops improving
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch + 1} (loss plateau)")
                break

    # Save model + class names
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "class_names": class_names,
    }, MODEL_PATH)

    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)

    # Invalidate cached model so next predict_name() loads the fresh one
    # Record RAW sample count (pre-augmentation) so needs_training() compares apples-to-apples
    global _model, _classes, _last_train_count
    with _model_lock:
        _model = None
        _classes = None
        stats = get_training_stats()
        _last_train_count = sum(stats.values())

    # Final training accuracy
    model.eval()
    with torch.no_grad():
        all_out = model(X.to(device))
        all_pred = all_out.argmax(dim=1)
        train_acc = float((all_pred == y.to(device)).sum()) / len(y)

    result = {
        "accuracy": train_acc,
        "val_accuracy": best_acc,
        "num_classes": num_classes,
        "num_samples": len(images),
        "class_names": class_names,
        "final_loss": epoch_losses[-1],
    }
    print(f"[name_model] Training complete: acc={train_acc:.3f} val_acc={best_acc:.3f}")
    print(f"[name_model] Model saved: {MODEL_PATH}")
    return result


# ── Prediction ────────────────────────────────────────────────────────────────
def _load_model():
    """Load the trained model (lazy, cached). Thread-safe."""
    global _model, _classes, _device

    with _model_lock:
        if _model is not None:
            return _model, _classes

        if not os.path.isfile(MODEL_PATH):
            return None, None

        try:
            import torch
            _device = torch.device("cpu")
            checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)
            _classes = checkpoint["class_names"]
            _model = _build_model(checkpoint["num_classes"]).to(_device)
            _model.load_state_dict(checkpoint["state_dict"])
            _model.eval()
            print(f"[name_model] Loaded model: {len(_classes)} classes")
            return _model, _classes
        except Exception as e:
            print(f"[name_model] Failed to load model: {e}")
            return None, None


def predict_name(shot) -> tuple:
    """Predict the player name from a name-region screenshot.

    Args:
        shot: PIL Image of the raw name region

    Returns:
        (player_name: str, confidence: float). ('', 0.0) if model not available.
    """
    model, classes = _load_model()
    if model is None:
        return '', 0.0

    try:
        import torch

        gray = shot.convert("L").resize((IMG_W, IMG_H), Img.LANCZOS)
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(_device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            player = classes[idx.item()]
            confidence = conf.item()

        return player, confidence
    except Exception as e:
        print(f"[name_model] Prediction error: {e}")
        return '', 0.0


def is_model_ready() -> bool:
    """Check if a trained model file exists."""
    return os.path.isfile(MODEL_PATH)
