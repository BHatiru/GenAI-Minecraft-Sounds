"""
Compute evaluation metrics for AudioGen LoRA fine-tuning:
1. CLAP cosine similarity (text-audio alignment)
2. Spectral centroid / bandwidth comparison
3. Log-mel spectrogram visualizations
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs" / "audiogen"
DATA_DIR = ROOT / "data" / "processed"
PRES_DIR = ROOT / "presentation"

# ─── 1. Gather audio files ───────────────────────────────────────
baseline_dir = OUTPUTS / "baseline"
lora_dir = OUTPUTS / "lora"
gen_dir = OUTPUTS / "generalization"

baseline_wavs = sorted(baseline_dir.glob("*.wav"))
lora_wavs = sorted(lora_dir.glob("*.wav"))
gen_wavs = sorted(gen_dir.glob("*.wav"))

# Reference Minecraft sounds
ref_wavs = sorted(DATA_DIR.rglob("*.wav"))[:50]  # sample up to 50

print(f"Baseline: {len(baseline_wavs)}, LoRA: {len(lora_wavs)}, "
      f"Generalization: {len(gen_wavs)}, Reference: {len(ref_wavs)}")

# ─── 2. Spectral Features ────────────────────────────────────────
def load_audio(wav_path, target_sr=16000):
    """Load audio as torch tensor [1, T] at target_sr."""
    data, sr = sf.read(str(wav_path), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.T  # [channels, T]
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def compute_spectral_features(wav_path, target_sr=16000):
    """Compute spectral centroid, bandwidth, and RMS energy."""
    wav, sr = load_audio(wav_path, target_sr)
    wav = wav.mean(0, keepdim=True)  # mono

    # Spectral centroid
    spec = torch.stft(wav.squeeze(), n_fft=1024, hop_length=512,
                      return_complex=True, window=torch.hann_window(1024))
    mag = spec.abs()
    freqs = torch.linspace(0, sr/2, mag.shape[0])
    centroid = (freqs.unsqueeze(1) * mag).sum(0) / (mag.sum(0) + 1e-8)

    # RMS energy
    rms = wav.pow(2).mean().sqrt().item()

    return {
        "centroid_mean": centroid.mean().item(),
        "centroid_std": centroid.std().item(),
        "rms": rms,
        "duration_s": wav.shape[-1] / sr,
    }


def avg_features(wav_list):
    feats = [compute_spectral_features(w) for w in wav_list]
    keys = feats[0].keys()
    return {k: np.mean([f[k] for f in feats]) for k in keys}


print("\n=== Spectral Features ===")
ref_feats = avg_features(ref_wavs)
base_feats = avg_features(baseline_wavs)
lora_feats = avg_features(lora_wavs)

print(f"{'':20s} {'Centroid':>10s} {'Cntrd Std':>10s} {'RMS':>8s}")
print(f"{'Reference':20s} {ref_feats['centroid_mean']:10.1f} {ref_feats['centroid_std']:10.1f} {ref_feats['rms']:8.4f}")
print(f"{'Baseline (vanilla)':20s} {base_feats['centroid_mean']:10.1f} {base_feats['centroid_std']:10.1f} {base_feats['rms']:8.4f}")
print(f"{'LoRA (ours)':20s} {lora_feats['centroid_mean']:10.1f} {lora_feats['centroid_std']:10.1f} {lora_feats['rms']:8.4f}")

# ─── 3. CLAP similarity ──────────────────────────────────────────
print("\n=== CLAP Cosine Similarity ===")
try:
    from transformers import ClapModel, ClapProcessor
    clap_model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
    clap_proc = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
    clap_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_model = clap_model.to(device)

    def clap_text_audio_sim(text, wav_path):
        wav, sr = load_audio(wav_path, 48000)
        wav = wav.squeeze().numpy()
        inputs = clap_proc(text=[text], audio=[wav], return_tensors="pt",
                           sampling_rate=48000, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = clap_model(**inputs)
            text_emb = out.text_embeds
            audio_emb = out.audio_embeds
            sim = torch.nn.functional.cosine_similarity(text_emb, audio_emb).item()
        return sim

    # Prompts for baseline / lora
    prompts_4 = [
        "minecraft zombie groaning in a dark cave",
        "minecraft skeleton shooting arrows at player",
        "minecraft creeper hissing and exploding",
        "minecraft footsteps walking on stone blocks",
    ]

    # Match prompts to file pairs
    base_sims, lora_sims = [], []
    for i, prompt in enumerate(prompts_4):
        bfiles = [f for f in baseline_wavs if f"prompt{i+1}" in f.name or f"p{i+1}" in f.name]
        lfiles = [f for f in lora_wavs if f"prompt{i+1}" in f.name or f"p{i+1}" in f.name]

        # Fallback: pair by sorted order
        if not bfiles:
            idx = i * 2
            bfiles = baseline_wavs[idx:idx+2] if idx < len(baseline_wavs) else []
        if not lfiles:
            idx = i * 2
            lfiles = lora_wavs[idx:idx+2] if idx < len(lora_wavs) else []

        for f in bfiles:
            s = clap_text_audio_sim(prompt, f)
            base_sims.append(s)
            print(f"  Baseline {f.name}: {s:.3f}  ({prompt[:40]}...)")
        for f in lfiles:
            s = clap_text_audio_sim(prompt, f)
            lora_sims.append(s)
            print(f"  LoRA     {f.name}: {s:.3f}  ({prompt[:40]}...)")

    # Generalization prompts
    gen_lora_sims = []
    gen_vanilla_sims = []
    gen_prompts_map = {
        "minecraft_zombie_groaning_in_a_dark_cave": "minecraft zombie groaning in a dark cave",
        "creeper_hiss_then_explosion_player_hurt_": "creeper hiss then explosion, player hurt sound",
        "footsteps_on_stone_then_skeleton_arrow_s": "footsteps on stone then skeleton arrow shooting",
        "cave_ambience_with_water_dripping_and_di": "cave ambience with water dripping and distant mobs",
        "blaze_fireball_whoosh_impact_explosion_p": "blaze fireball whoosh impact explosion",
        "skeleton_hurt_ghast_moan_sound_player_ta": "skeleton hurt, ghast moan sound, player take damage",
    }

    for gf in gen_wavs:
        name = gf.stem
        is_lora = name.endswith("_lora")
        is_vanilla = name.endswith("_vanilla")
        # Find matching prompt
        matched_prompt = None
        for prefix, prompt in gen_prompts_map.items():
            if name.startswith(prefix):
                matched_prompt = prompt
                break
        if matched_prompt is None:
            # Try the comma version
            if "skeleton_hurt," in name:
                matched_prompt = "skeleton hurt, ghast moan sound, player take damage"
            else:
                continue
        sim = clap_text_audio_sim(matched_prompt, gf)
        if is_lora:
            gen_lora_sims.append(sim)
        elif is_vanilla:
            gen_vanilla_sims.append(sim)

    print(f"\n--- CLAP Summary ---")
    print(f"Baseline avg:     {np.mean(base_sims):.3f} ± {np.std(base_sims):.3f}")
    print(f"LoRA avg:         {np.mean(lora_sims):.3f} ± {np.std(lora_sims):.3f}")
    if gen_lora_sims:
        print(f"Gen LoRA avg:     {np.mean(gen_lora_sims):.3f} ± {np.std(gen_lora_sims):.3f}")
        print(f"Gen Vanilla avg:  {np.mean(gen_vanilla_sims):.3f} ± {np.std(gen_vanilla_sims):.3f}")

    HAVE_CLAP = True
except Exception as e:
    print(f"CLAP not available: {e}")
    base_sims = lora_sims = gen_lora_sims = gen_vanilla_sims = []
    HAVE_CLAP = False

# ─── 4. Visualization: spectrograms ──────────────────────────────
print("\n=== Generating Visualizations ===")

def plot_mel_spectrogram(wav_path, ax, title, sr=16000):
    wav, _ = load_audio(wav_path, sr)
    wav = wav.mean(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=512, n_mels=80)
    mel = mel_transform(wav)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
    ax.imshow(mel_db.numpy(), aspect="auto", origin="lower",
              cmap="magma", vmin=-80, vmax=0)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Frame")

# Compare baseline vs LoRA for first 4 prompts
fig, axes = plt.subplots(4, 2, figsize=(12, 10))
fig.suptitle("Mel Spectrogram: Baseline (vanilla) vs LoRA", fontsize=13, fontweight="bold")
for i in range(min(4, len(baseline_wavs))):
    plot_mel_spectrogram(baseline_wavs[i], axes[i, 0],
                         f"Baseline: {baseline_wavs[i].stem[:40]}")
    if i < len(lora_wavs):
        plot_mel_spectrogram(lora_wavs[i], axes[i, 1],
                             f"LoRA: {lora_wavs[i].stem[:40]}")
plt.tight_layout()
plt.savefig(PRES_DIR / "spectrograms_comparison.png", dpi=150)
plt.close()
print(f"Saved: spectrograms_comparison.png")

# Generalization spectrograms (pick one duration set)
gen_lora_only = [f for f in gen_wavs if f.stem.endswith("_lora")]
if len(gen_lora_only) >= 4:
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    fig.suptitle("Generalization: LoRA Outputs (Novel/Combo Prompts)", fontsize=13, fontweight="bold")
    for idx, ax in enumerate(axes.flat):
        if idx < len(gen_lora_only):
            plot_mel_spectrogram(gen_lora_only[idx], ax,
                                 gen_lora_only[idx].stem[:45])
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(PRES_DIR / "generalization_spectrograms.png", dpi=150)
    plt.close()
    print(f"Saved: generalization_spectrograms.png")

# ─── 5. CLAP bar chart ───────────────────────────────────────────
if HAVE_CLAP and base_sims and lora_sims:
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["In-Domain\n(Baseline)", "In-Domain\n(LoRA)"]
    means = [np.mean(base_sims), np.mean(lora_sims)]
    stds = [np.std(base_sims), np.std(lora_sims)]
    if gen_vanilla_sims and gen_lora_sims:
        categories += ["Generalization\n(Baseline)", "Generalization\n(LoRA)"]
        means += [np.mean(gen_vanilla_sims), np.mean(gen_lora_sims)]
        stds += [np.std(gen_vanilla_sims), np.std(gen_lora_sims)]
    colors = ["#6baed6", "#2171b5", "#fdae6b", "#e6550d"][:len(categories)]
    bars = ax.bar(categories, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
    ax.set_ylabel("CLAP Cosine Similarity")
    ax.set_title("Text-Audio Alignment: CLAP Cosine Similarity", fontweight="bold")
    ax.set_ylim(0, max(means) * 1.3 + 0.05)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PRES_DIR / "clap_similarity.png", dpi=150)
    plt.close()
    print(f"Saved: clap_similarity.png")

# ─── 6. Spectral comparison chart ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
labels = ["Reference\n(Real MC)", "Baseline\n(Vanilla)", "LoRA\n(Ours)"]
centroids = [ref_feats["centroid_mean"], base_feats["centroid_mean"], lora_feats["centroid_mean"]]
rms_vals = [ref_feats["rms"], base_feats["rms"], lora_feats["rms"]]
colors3 = ["#2ca02c", "#6baed6", "#e6550d"]

axes[0].bar(labels, centroids, color=colors3, edgecolor="black")
axes[0].set_ylabel("Hz")
axes[0].set_title("Mean Spectral Centroid", fontweight="bold")
for i, v in enumerate(centroids):
    axes[0].text(i, v + 20, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

axes[1].bar(labels, rms_vals, color=colors3, edgecolor="black")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Mean RMS Energy", fontweight="bold")
for i, v in enumerate(rms_vals):
    axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

plt.suptitle("Spectral Feature Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PRES_DIR / "spectral_comparison.png", dpi=150)
plt.close()
print(f"Saved: spectral_comparison.png")

# ─── 7. Save all metrics to JSON ─────────────────────────────────
metrics = {
    "spectral": {
        "reference": ref_feats,
        "baseline": base_feats,
        "lora": lora_feats,
    },
    "clap": {
        "baseline_scores": base_sims,
        "lora_scores": lora_sims,
        "gen_lora_scores": gen_lora_sims,
        "gen_vanilla_scores": gen_vanilla_sims,
        "baseline_mean": float(np.mean(base_sims)) if base_sims else None,
        "lora_mean": float(np.mean(lora_sims)) if lora_sims else None,
        "gen_lora_mean": float(np.mean(gen_lora_sims)) if gen_lora_sims else None,
        "gen_vanilla_mean": float(np.mean(gen_vanilla_sims)) if gen_vanilla_sims else None,
    },
    "outputs": {
        "baseline_count": len(baseline_wavs),
        "lora_count": len(lora_wavs),
        "generalization_count": len(gen_wavs),
        "checkpoints": ["best", "epoch_0005", "epoch_0010", "epoch_0015",
                        "epoch_0020", "epoch_0025", "final"],
    },
    "training": {
        "epochs": 25,
        "dataset_train": 277,
        "dataset_val": 48,
        "lora_rank": 128,
        "lora_alpha": 256,
        "trainable_params_M": 132,
        "optimizer": "AdamW",
        "lr": 1e-4,
    }
}

with open(PRES_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved: metrics.json")
print("\n=== Done ===")
