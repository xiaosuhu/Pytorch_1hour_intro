"""
Generate small synthetic 2D "spectrogram-like" data for PyTorch intro workshop.
Mimics the shape/structure of the real HMS 2D CNN pipeline (4 bipolar montage
channels, 6-class classification) WITHOUT using any real competition data —
avoids Kaggle redistribution restrictions.

Kept intentionally small (small image size, float16, few samples) so the
resulting file is a few hundred KB and downloads instantly in a 1-hour
Colab workshop. Each class gets a distinct synthetic frequency-band
signature so a simple CNN can actually learn something visible during
the session (loss goes down, accuracy goes up).
"""
import numpy as np

np.random.seed(42)

LABELS = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
N_CHANNELS = 4      # LL, RL, LP, RP (matches real bipolar montage structure)
N_FREQ = 32          # downsized freq bins (real pipeline uses 100)
N_TIME = 64          # downsized time steps (real pipeline uses 300)
N_PER_CLASS_TRAIN = 25   # 25*6 = 150 train samples
N_PER_CLASS_VAL = 8      # 8*6 = 48 val samples

CLASS_SIGNATURES = {
    0: dict(freq_center=5,  freq_width=3, time_pattern="pulse"),    # Seizure - narrow, high-freq bursts
    1: dict(freq_center=12, freq_width=5, time_pattern="steady"),   # LPD - mid-freq, steady
    2: dict(freq_center=20, freq_width=5, time_pattern="steady"),   # GPD - higher-freq, steady
    3: dict(freq_center=8,  freq_width=4, time_pattern="rhythmic"), # LRDA - rhythmic
    4: dict(freq_center=16, freq_width=6, time_pattern="rhythmic"), # GRDA - rhythmic, broader
    5: dict(freq_center=None, freq_width=None, time_pattern="noise"),# Other - just noise
}

def make_sample(label):
    sig = CLASS_SIGNATURES[label]
    img = np.random.randn(N_CHANNELS, N_FREQ, N_TIME).astype(np.float32) * 0.9  # more noise -> needs a few epochs to learn

    if sig["time_pattern"] == "noise":
        return img.astype(np.float16)

    freq_axis = np.arange(N_FREQ)[:, None]
    band = np.exp(-0.5 * ((freq_axis - sig["freq_center"]) / sig["freq_width"]) ** 2)

    time_axis = np.arange(N_TIME)[None, :]
    if sig["time_pattern"] == "pulse":
        modulation = np.zeros((1, N_TIME))
        n_pulses = np.random.randint(2, 4)
        centers = np.random.choice(N_TIME, n_pulses, replace=False)
        for c in centers:
            modulation += np.exp(-0.5 * ((time_axis - c) / 3) ** 2)
        modulation = np.clip(modulation, 0, 1)
    elif sig["time_pattern"] == "steady":
        modulation = np.ones((1, N_TIME)) * 0.8
    elif sig["time_pattern"] == "rhythmic":
        freq_hz = np.random.uniform(0.5, 1.5)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * freq_hz * time_axis / N_TIME * 6)
    else:
        modulation = np.zeros((1, N_TIME))

    pattern = band @ modulation

    for ch in range(N_CHANNELS):
        ch_scale = np.random.uniform(0.7, 1.3)
        img[ch] += pattern * 1.2 * ch_scale  # weaker signal relative to noise

    return img.astype(np.float16)


def build_split(n_per_class):
    X, y = [], []
    for label in range(6):
        for _ in range(n_per_class):
            X.append(make_sample(label))
            y.append(label)
    X = np.stack(X).astype(np.float16)
    y = np.array(y, dtype=np.int64)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

X_train, y_train = build_split(N_PER_CLASS_TRAIN)
X_val, y_val = build_split(N_PER_CLASS_VAL)

out_path = "/home/claude/pytorch_demo/pytorch_demo_spectrograms.npz"
np.savez_compressed(
    out_path,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    label_names=np.array(LABELS),
)

import os
size_kb = os.path.getsize(out_path) / 1024
print(f"Saved: {out_path}")
print(f"File size: {size_kb:.1f} KB")
print(f"X_train shape: {X_train.shape} dtype={X_train.dtype}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"Label distribution (train): {np.bincount(y_train)}")
print(f"Labels: {LABELS}")
