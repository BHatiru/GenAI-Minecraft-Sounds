#!/usr/bin/env python3
"""Generate presentation visuals from processed audio data."""
import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

OUT = Path(__file__).parent
DATA = Path(__file__).resolve().parent.parent / "data" / "processed"

plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#1e1e2e",
    "text.color": "#cdd6f4",
    "axes.labelcolor": "#cdd6f4",
    "xtick.color": "#a6adc8",
    "ytick.color": "#a6adc8",
    "font.size": 11,
})

# ── 1. Spectrogram grid: 4 diverse samples ──────────────────────────
SAMPLES = [
    ("mob/zombie/hurt_death.wav",       "Zombie Hurt → Death"),
    ("mob/ghast/scream_seq.wav",        "Ghast Scream Sequence"),
    ("step/stone_walk.wav",             "Stone Footsteps (Walk)"),
    ("ambient/cave_slow.wav",           "Cave Ambient (Slow)"),
]

# fallback if exact files don't exist
def find_samples():
    found = []
    for rel, label in SAMPLES:
        p = DATA / rel
        if p.exists():
            found.append((p, label))
        else:
            # find any .wav in that category
            cat = DATA / Path(rel).parent
            if cat.exists():
                wavs = sorted(cat.glob("*.wav"))
                if wavs:
                    name = wavs[0].stem.replace("_", " ").title()
                    found.append((wavs[0], name))
    # pad with random extras if needed
    if len(found) < 4:
        for d in sorted(DATA.rglob("*.wav")):
            if d not in [f[0] for f in found]:
                found.append((d, d.stem.replace("_", " ").title()))
            if len(found) >= 4:
                break
    return found[:4]


def plot_spectrograms():
    samples = find_samples()
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    fig.suptitle("Processed Minecraft Sound Samples — Mel Spectrograms",
                 fontsize=15, fontweight="bold", color="#cdd6f4")

    for ax, (path, label) in zip(axes.flat, samples):
        y, sr = sf.read(str(path), dtype="float32")
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                           hop_length=160, n_mels=64,
                                           fmin=0, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, hop_length=160,
                                 x_axis="time", y_axis="mel",
                                 ax=ax, cmap="magma")
        ax.set_title(label, fontsize=11, color="#cdd6f4")
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT / "spectrogram_samples.png"
    fig.savefig(str(out), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── 2. Dataset composition bar chart ────────────────────────────────
def plot_dataset_composition():
    categories = {}
    for cat_dir in sorted(DATA.iterdir()):
        if cat_dir.is_dir():
            count = len(list(cat_dir.rglob("*.wav")))
            if count > 0:
                categories[cat_dir.name.title()] = count

    cats = list(categories.keys())
    counts = list(categories.values())
    colors = ["#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#89dceb",
              "#b4befe", "#cba6f7", "#f5c2e7"]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(cats, counts, color=colors[:len(cats)], edgecolor="#313244")
    ax.set_xlabel("Number of Clips")
    ax.set_title("Dataset Composition by Category",
                 fontsize=14, fontweight="bold", color="#cdd6f4")
    ax.invert_yaxis()

    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(c), va="center", color="#cdd6f4", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#585b70")
    ax.spines["left"].set_color("#585b70")

    total = sum(counts)
    ax.text(0.98, 0.02, f"Total: {total} clips",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12, color="#a6e3a1", fontweight="bold")

    plt.tight_layout()
    out = OUT / "dataset_composition.png"
    fig.savefig(str(out), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── 3. Waveform + spectrogram side-by-side for one sample ───────────
def plot_waveform_and_spec():
    # Pick a zombie sample for recognizability
    sample = DATA / "mob" / "zombie" / "hurt_death.wav"
    if not sample.exists():
        wavs = sorted((DATA / "mob").rglob("*.wav"))
        sample = wavs[0] if wavs else sorted(DATA.rglob("*.wav"))[0]

    y, sr = sf.read(str(sample), dtype="float32")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.5))
    fig.suptitle(f"Sample: {sample.parent.name}/{sample.name}  |  16 kHz · 4s · mono",
                 fontsize=13, fontweight="bold", color="#cdd6f4")

    # Waveform
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color="#89b4fa", linewidth=0.4)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Waveform", color="#cdd6f4")
    ax1.set_xlim(0, len(y)/sr)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_color("#585b70")
    ax1.spines["left"].set_color("#585b70")

    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                       hop_length=160, n_mels=64,
                                       fmin=0, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=160,
                             x_axis="time", y_axis="mel",
                             ax=ax2, cmap="magma")
    ax2.set_title("Mel Spectrogram", color="#cdd6f4")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = OUT / "waveform_spectrogram.png"
    fig.savefig(str(out), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_spectrograms()
    plot_dataset_composition()
    plot_waveform_and_spec()
    print("All visuals generated.")
