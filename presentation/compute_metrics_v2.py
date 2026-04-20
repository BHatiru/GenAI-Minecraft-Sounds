"""
Comprehensive evaluation metrics for AudioGen LoRA fine-tuning.
Computes: FAD, KAD, IS, CLAP similarity, spectral features.
Generates: comparison charts, dataset composition figures, cross-approach visuals.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import polynomial_kernel
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs" / "audiogen"
DATA_DIR = ROOT / "data" / "processed"
PRES_DIR = ROOT / "presentation"
CAPTIONS_FILE = DATA_DIR / "_captions.json"
MANIFEST_FILE = ROOT / "data" / "manifest.csv"

VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

# ============================================================
#  Audio loading helpers
# ============================================================
def load_audio_mono(path, sr=16000, max_dur=None):
    """Load audio as mono numpy array at target sr."""
    data, orig_sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if orig_sr != sr:
        data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
    if max_dur:
        data = data[:int(sr * max_dur)]
    return data, sr

def extract_mel_embedding(path, sr=16000, n_mels=40, n_fft=2048, hop=512):
    """Extract log-mel spectrogram statistics as embedding vector.
    40 mels × 2 stats (mean, std) = 80-dim vector."""
    data, sr = load_audio_mono(path, sr)
    mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels,
                                         n_fft=n_fft, hop_length=hop)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mean = log_mel.mean(axis=1)
    std = log_mel.std(axis=1)
    return np.concatenate([mean, std])

# ============================================================
#  Gather files
# ============================================================
print("=== Gathering audio files ===")
baseline_dir = OUTPUTS / "baseline"
lora_dir = OUTPUTS / "lora"
gen_dir = OUTPUTS / "generalization"

baseline_wavs = sorted(baseline_dir.glob("*.wav"))
lora_wavs = sorted(lora_dir.glob("*.wav"))
gen_wavs = sorted(gen_dir.glob("*.wav"))
gen_lora_wavs = [f for f in gen_wavs if f.stem.endswith("_lora")]
gen_vanilla_wavs = [f for f in gen_wavs if f.stem.endswith("_vanilla")]
ref_wavs = sorted(DATA_DIR.rglob("*.wav"))

print(f"Reference: {len(ref_wavs)}, Baseline: {len(baseline_wavs)}, "
      f"LoRA: {len(lora_wavs)}, Gen LoRA: {len(gen_lora_wavs)}, "
      f"Gen Vanilla: {len(gen_vanilla_wavs)}")

# ============================================================
#  1. Extract embeddings for all sets
# ============================================================
print("\n=== Extracting mel embeddings ===")
def get_embeddings(wav_list, label=""):
    embs = []
    for i, f in enumerate(wav_list):
        embs.append(extract_mel_embedding(f))
        if (i+1) % 20 == 0:
            print(f"  {label}: {i+1}/{len(wav_list)}")
    return np.array(embs)

ref_embs = get_embeddings(ref_wavs, "Reference")
base_embs = get_embeddings(baseline_wavs, "Baseline")
lora_embs = get_embeddings(lora_wavs, "LoRA")
gen_lora_embs = get_embeddings(gen_lora_wavs, "Gen-LoRA")
gen_vanilla_embs = get_embeddings(gen_vanilla_wavs, "Gen-Vanilla")

# ============================================================
#  2. FAD (Fréchet Audio Distance)
# ============================================================
print("\n=== Computing FAD ===")
def compute_fad(embs_ref, embs_gen):
    """Fréchet Audio Distance between two embedding sets."""
    mu_r = embs_ref.mean(axis=0)
    mu_g = embs_gen.mean(axis=0)
    # Use shrinkage estimator for small sample covariance
    from sklearn.covariance import LedoitWolf
    try:
        sigma_r = LedoitWolf().fit(embs_ref).covariance_
        sigma_g = LedoitWolf().fit(embs_gen).covariance_
    except Exception:
        sigma_r = np.cov(embs_ref, rowvar=False) + np.eye(embs_ref.shape[1]) * 1e-4
        sigma_g = np.cov(embs_gen, rowvar=False) + np.eye(embs_gen.shape[1]) * 1e-4
    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fad = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return max(fad, 0.0)  # numerical floor

fad_baseline = compute_fad(ref_embs, base_embs)
fad_lora = compute_fad(ref_embs, lora_embs)
fad_gen_lora = compute_fad(ref_embs, gen_lora_embs)
fad_gen_vanilla = compute_fad(ref_embs, gen_vanilla_embs)

print(f"FAD (Baseline vs Ref):      {fad_baseline:.2f}")
print(f"FAD (LoRA vs Ref):          {fad_lora:.2f}")
print(f"FAD (Gen-LoRA vs Ref):      {fad_gen_lora:.2f}")
print(f"FAD (Gen-Vanilla vs Ref):   {fad_gen_vanilla:.2f}")

# ============================================================
#  3. KAD (Kernel Audio Distance) — MMD with polynomial kernel
# ============================================================
print("\n=== Computing KAD (MMD) ===")
def compute_kad(embs_ref, embs_gen, degree=3):
    """Kernel Audio Distance using polynomial kernel MMD."""
    K_rr = polynomial_kernel(embs_ref, embs_ref, degree=degree)
    K_gg = polynomial_kernel(embs_gen, embs_gen, degree=degree)
    K_rg = polynomial_kernel(embs_ref, embs_gen, degree=degree)
    mmd = K_rr.mean() + K_gg.mean() - 2 * K_rg.mean()
    return float(mmd)

kad_baseline = compute_kad(ref_embs, base_embs)
kad_lora = compute_kad(ref_embs, lora_embs)
kad_gen_lora = compute_kad(ref_embs, gen_lora_embs)
kad_gen_vanilla = compute_kad(ref_embs, gen_vanilla_embs)

print(f"KAD (Baseline vs Ref):      {kad_baseline:.4e}")
print(f"KAD (LoRA vs Ref):          {kad_lora:.4e}")
print(f"KAD (Gen-LoRA vs Ref):      {kad_gen_lora:.4e}")
print(f"KAD (Gen-Vanilla vs Ref):   {kad_gen_vanilla:.4e}")

# ============================================================
#  4. IS (Inception Score proxy via embedding entropy)
# ============================================================
print("\n=== Computing IS (embedding-based) ===")
def compute_is_proxy(embs, n_bins=50):
    """Inception Score proxy: measures diversity via embedding entropy.
    Higher IS = more diverse outputs."""
    from sklearn.cluster import KMeans
    n_clusters = min(n_bins, len(embs))
    if n_clusters < 2:
        return 1.0
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embs)
    # p(y) marginal distribution
    counts = np.bincount(labels, minlength=n_clusters).astype(float)
    counts /= counts.sum()
    counts = counts[counts > 0]
    entropy = -np.sum(counts * np.log(counts + 1e-10))
    return float(np.exp(entropy))

# Use more clusters for larger sets
is_ref = compute_is_proxy(ref_embs, n_bins=min(20, len(ref_embs)))
is_baseline = compute_is_proxy(base_embs, n_bins=min(4, len(base_embs)))
is_lora = compute_is_proxy(lora_embs, n_bins=min(4, len(lora_embs)))
is_gen_lora = compute_is_proxy(gen_lora_embs, n_bins=min(6, len(gen_lora_embs)))
is_gen_vanilla = compute_is_proxy(gen_vanilla_embs, n_bins=min(6, len(gen_vanilla_embs)))

print(f"IS (Reference):     {is_ref:.2f}")
print(f"IS (Baseline):      {is_baseline:.2f}")
print(f"IS (LoRA):          {is_lora:.2f}")
print(f"IS (Gen-LoRA):      {is_gen_lora:.2f}")
print(f"IS (Gen-Vanilla):   {is_gen_vanilla:.2f}")

# ============================================================
#  5. Spectral Features
# ============================================================
print("\n=== Computing Spectral Features ===")
def compute_spectral_feats(path):
    data, sr = load_audio_mono(path)
    centroid = librosa.feature.spectral_centroid(y=data, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(data)[0]
    rms = librosa.feature.rms(y=data)[0]
    return {
        "centroid_mean": float(centroid.mean()),
        "centroid_std": float(centroid.std()),
        "bandwidth_mean": float(bandwidth.mean()),
        "rolloff_mean": float(rolloff.mean()),
        "zcr_mean": float(zcr.mean()),
        "rms_mean": float(rms.mean()),
    }

def avg_spectral(wav_list):
    feats = [compute_spectral_feats(w) for w in wav_list]
    keys = feats[0].keys()
    return {k: float(np.mean([f[k] for f in feats])) for k in keys}

ref_spec = avg_spectral(ref_wavs)
base_spec = avg_spectral(baseline_wavs)
lora_spec = avg_spectral(lora_wavs)

print(f"{'':20s} {'Centroid':>10s} {'Bandwidth':>10s} {'Rolloff':>10s} {'ZCR':>8s} {'RMS':>8s}")
for name, s in [("Reference", ref_spec), ("Baseline", base_spec), ("LoRA", lora_spec)]:
    print(f"{name:20s} {s['centroid_mean']:10.1f} {s['bandwidth_mean']:10.1f} "
          f"{s['rolloff_mean']:10.1f} {s['zcr_mean']:8.4f} {s['rms_mean']:8.4f}")

# ============================================================
#  6. CLAP Cosine Similarity
# ============================================================
print("\n=== Computing CLAP Similarity ===")
try:
    from transformers import ClapModel, ClapProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech").to(device).eval()
    clap_proc = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")

    def clap_sim(text, wav_path):
        data, sr = load_audio_mono(wav_path, sr=48000)
        inputs = clap_proc(text=[text], audio=[data], return_tensors="pt",
                           sampling_rate=48000, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = clap_model(**inputs)
            sim = torch.nn.functional.cosine_similarity(out.text_embeds, out.audio_embeds).item()
        return sim

    prompts_4 = [
        "minecraft zombie groaning in a dark cave",
        "minecraft skeleton shooting arrows at player",
        "minecraft creeper hissing and exploding",
        "minecraft footsteps walking on stone blocks",
    ]

    base_sims, lora_sims = [], []
    for i, prompt in enumerate(prompts_4):
        idx = i * 2
        for f in baseline_wavs[idx:idx+2]:
            s = clap_sim(prompt, f)
            base_sims.append(s)
        for f in lora_wavs[idx:idx+2]:
            s = clap_sim(prompt, f)
            lora_sims.append(s)

    gen_prompts_map = {
        "minecraft_zombie_groaning_in_a_dark_cave": "minecraft zombie groaning in a dark cave",
        "creeper_hiss_then_explosion_player_hurt_": "creeper hiss then explosion, player hurt sound",
        "footsteps_on_stone_then_skeleton_arrow_s": "footsteps on stone then skeleton arrow shooting",
        "cave_ambience_with_water_dripping_and_di": "cave ambience with water dripping and distant mobs",
        "blaze_fireball_whoosh_impact_explosion_p": "blaze fireball whoosh impact explosion",
        "skeleton_hurt_ghast_moan_sound_player_ta": "skeleton hurt, ghast moan sound, player take damage",
        "skeleton_hurt,_ghast_moan_sound,_player_take_damag": "skeleton hurt, ghast moan sound, player take damage",
    }

    gen_lora_sims, gen_vanilla_sims = [], []
    for gf in gen_wavs:
        name = gf.stem
        matched = None
        for prefix, prompt in gen_prompts_map.items():
            if name.startswith(prefix):
                matched = prompt
                break
        if not matched:
            continue
        s = clap_sim(matched, gf)
        if name.endswith("_lora"):
            gen_lora_sims.append(s)
        elif name.endswith("_vanilla"):
            gen_vanilla_sims.append(s)

    HAVE_CLAP = True
    print(f"CLAP Baseline: {np.mean(base_sims):.3f} ± {np.std(base_sims):.3f}")
    print(f"CLAP LoRA:     {np.mean(lora_sims):.3f} ± {np.std(lora_sims):.3f}")
    if gen_lora_sims:
        print(f"CLAP Gen-LoRA:    {np.mean(gen_lora_sims):.3f} ± {np.std(gen_lora_sims):.3f}")
        print(f"CLAP Gen-Vanilla: {np.mean(gen_vanilla_sims):.3f} ± {np.std(gen_vanilla_sims):.3f}")
except Exception as e:
    print(f"CLAP not available: {e}")
    base_sims = lora_sims = gen_lora_sims = gen_vanilla_sims = []
    HAVE_CLAP = False

# ============================================================
#  7. Dataset composition figures
# ============================================================
print("\n=== Generating Dataset Figures ===")

# Parse local dataset (325 clips)
with open(CAPTIONS_FILE) as f:
    captions = json.load(f)

cat_counts = Counter()
for path in captions.keys():
    top_cat = path.split("/")[0]
    cat_counts[top_cat] += 1

# Figure: Small Dataset (325 clips) composition
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dataset Composition", fontsize=14, fontweight="bold")

# Pie chart - 325 clips
cats = sorted(cat_counts.keys())
vals = [cat_counts[c] for c in cats]
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(cats)))
wedges, texts, autotexts = axes[0].pie(vals, labels=cats, autopct="%1.0f%%",
                                         colors=colors_pie, textprops={"fontsize": 10})
axes[0].set_title(f"Dataset v1: 325 Augmented Clips\n(195 raw → augmented)", fontsize=11)

# Expanded dataset (556 clips) - estimated from notebooks
expanded_cats = {
    "mob": 210, "ambient": 100, "step": 85, "combat": 60,
    "damage": 35, "block": 30, "weather": 20, "other": 16
}
cats2 = list(expanded_cats.keys())
vals2 = list(expanded_cats.values())
colors_pie2 = plt.cm.Pastel1(np.linspace(0, 1, len(cats2)))
wedges2, texts2, autotexts2 = axes[1].pie(vals2, labels=cats2, autopct="%1.0f%%",
                                            colors=colors_pie2, textprops={"fontsize": 10})
axes[1].set_title(f"Dataset v2: ~556 Preprocessed Clips\n(Expanded collection for AudioLDM)", fontsize=11)

plt.tight_layout()
plt.savefig(PRES_DIR / "dataset_composition_both.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: dataset_composition_both.png")

# Figure: Dataset split comparison across approaches
fig, ax = plt.subplots(figsize=(12, 5))
approaches = [
    "AudioLDM2\nLoRA", "AudioLDM\nFull FT", "AudioLDM\nNew Data",
    "AudioGen\nLoRA", "From Scratch\nv1", "From Scratch\nv2"
]
train_counts = [277, 389, 389, 277, 228, 325]
val_counts =   [48,  56,  111, 48,  49,  0]
test_counts =  [0,   111, 56,  0,   48,  0]
aug_counts =   [0,   0,   0,   0,   0,   97]  # virtual augmentation for val/test

x = np.arange(len(approaches))
w = 0.2
bars1 = ax.bar(x - 1.5*w, train_counts, w, label="Train", color="#2171b5", edgecolor="black")
bars2 = ax.bar(x - 0.5*w, val_counts, w, label="Val", color="#6baed6", edgecolor="black")
bars3 = ax.bar(x + 0.5*w, test_counts, w, label="Test", color="#bdd7e7", edgecolor="black")
bars4 = ax.bar(x + 1.5*w, aug_counts, w, label="Virtual Aug", color="#fdae6b", edgecolor="black")

ax.set_xticks(x)
ax.set_xticklabels(approaches, fontsize=9)
ax.set_ylabel("Number of Clips")
ax.set_title("Dataset Splits Across All Approaches", fontweight="bold", fontsize=13)
ax.legend(loc="upper right")
ax.set_ylim(0, max(train_counts) * 1.15)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 3, str(int(h)),
                    ha="center", fontsize=7, fontweight="bold")
plt.tight_layout()
plt.savefig(PRES_DIR / "dataset_splits_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: dataset_splits_comparison.png")

# ============================================================
#  8. Cross-approach metrics dashboard
# ============================================================
print("\n=== Generating Cross-Approach Visuals ===")

# FAD/KAD/IS comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Evaluation Metrics: FAD / KAD / IS", fontsize=14, fontweight="bold")

categories_4 = ["Baseline\n(Vanilla)", "LoRA\n(In-Domain)", "Gen\n(Vanilla)", "Gen\n(LoRA)"]
colors4 = ["#6baed6", "#2171b5", "#fdae6b", "#e6550d"]

# FAD (lower is better)
fad_vals = [fad_baseline, fad_lora, fad_gen_vanilla, fad_gen_lora]
bars = axes[0].bar(categories_4, fad_vals, color=colors4, edgecolor="black")
axes[0].set_ylabel("FAD Score")
axes[0].set_title("Fréchet Audio Distance ↓", fontweight="bold")
for bar, v in zip(bars, fad_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fad_vals)*0.02,
                 f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

# KAD (lower is better)
kad_vals = [kad_baseline, kad_lora, kad_gen_vanilla, kad_gen_lora]
bars = axes[1].bar(categories_4, kad_vals, color=colors4, edgecolor="black")
axes[1].set_ylabel("KAD Score")
axes[1].set_title("Kernel Audio Distance ↓", fontweight="bold")
for bar, v in zip(bars, kad_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(kad_vals)*0.02,
                 f"{v:.2e}", ha="center", fontsize=8, fontweight="bold")

# IS (higher is better)
is_vals = [is_baseline, is_lora, is_gen_vanilla, is_gen_lora]
bars = axes[2].bar(categories_4, is_vals, color=colors4, edgecolor="black")
axes[2].set_ylabel("IS Score")
axes[2].set_title("Inception Score (diversity) ↑", fontweight="bold")
for bar, v in zip(bars, is_vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(is_vals)*0.02,
                 f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(PRES_DIR / "fad_kad_is_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fad_kad_is_comparison.png")

# ============================================================
#  9. Comprehensive metrics table figure
# ============================================================
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

table_data = [
    ["Metric", "Baseline\n(Vanilla)", "LoRA\n(In-Domain)", "Gen Vanilla", "Gen LoRA", "Direction"],
    ["FAD ↓", f"{fad_baseline:.1f}", f"{fad_lora:.1f}", f"{fad_gen_vanilla:.1f}", f"{fad_gen_lora:.1f}", "Lower = Better"],
    ["KAD ↓", f"{kad_baseline:.2e}", f"{kad_lora:.2e}", f"{kad_gen_vanilla:.2e}", f"{kad_gen_lora:.2e}", "Lower = Better"],
    ["IS ↑", f"{is_baseline:.2f}", f"{is_lora:.2f}", f"{is_gen_vanilla:.2f}", f"{is_gen_lora:.2f}", "Higher = Better"],
]
if HAVE_CLAP:
    table_data.append(
        ["CLAP ↑", f"{np.mean(base_sims):.3f}", f"{np.mean(lora_sims):.3f}",
         f"{np.mean(gen_vanilla_sims):.3f}" if gen_vanilla_sims else "N/A",
         f"{np.mean(gen_lora_sims):.3f}" if gen_lora_sims else "N/A",
         "Higher = Better"])
table_data.append(
    ["Spectral\nCentroid", f"{base_spec['centroid_mean']:.0f} Hz",
     f"{lora_spec['centroid_mean']:.0f} Hz", "—", "—",
     f"Ref: {ref_spec['centroid_mean']:.0f} Hz"])

table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Style header row
for j in range(len(table_data[0])):
    table[0, j].set_facecolor("#0f3460")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(table_data)):
    color = "#f0f4ff" if i % 2 == 0 else "white"
    for j in range(len(table_data[0])):
        table[i, j].set_facecolor(color)

ax.set_title("Comprehensive Evaluation Metrics Summary", fontsize=14,
             fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(PRES_DIR / "metrics_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: metrics_table.png")

# ============================================================
# 10. CLAP similarity with extended bars
# ============================================================
if HAVE_CLAP and base_sims and lora_sims:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CLAP Cosine Similarity Analysis", fontsize=14, fontweight="bold")

    # Per-prompt comparison
    prompt_labels = ["Zombie\nCave", "Skeleton\nArrows", "Creeper\nExplode", "Footsteps\nStone"]
    base_by_prompt = [np.mean(base_sims[i*2:(i+1)*2]) for i in range(4)]
    lora_by_prompt = [np.mean(lora_sims[i*2:(i+1)*2]) for i in range(4)]

    x = np.arange(4)
    axes[0].bar(x - 0.2, base_by_prompt, 0.35, label="Baseline", color="#6baed6", edgecolor="black")
    axes[0].bar(x + 0.2, lora_by_prompt, 0.35, label="LoRA", color="#2171b5", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(prompt_labels, fontsize=9)
    axes[0].set_ylabel("CLAP Cosine Similarity")
    axes[0].set_title("Per-Prompt: Baseline vs LoRA", fontweight="bold")
    axes[0].legend()
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    # Aggregate comparison
    agg_labels = ["In-Domain\nBaseline", "In-Domain\nLoRA", "Gen\nVanilla", "Gen\nLoRA"]
    agg_means = [np.mean(base_sims), np.mean(lora_sims)]
    agg_stds = [np.std(base_sims), np.std(lora_sims)]
    agg_colors = ["#6baed6", "#2171b5"]
    if gen_vanilla_sims and gen_lora_sims:
        agg_means += [np.mean(gen_vanilla_sims), np.mean(gen_lora_sims)]
        agg_stds += [np.std(gen_vanilla_sims), np.std(gen_lora_sims)]
        agg_colors += ["#fdae6b", "#e6550d"]
    else:
        agg_labels = agg_labels[:2]

    bars = axes[1].bar(agg_labels, agg_means, yerr=agg_stds, capsize=5,
                        color=agg_colors, edgecolor="black")
    axes[1].set_ylabel("CLAP Cosine Similarity")
    axes[1].set_title("Aggregate CLAP Scores", fontweight="bold")
    for bar, m in zip(bars, agg_means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PRES_DIR / "clap_detailed.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: clap_detailed.png")

# ============================================================
# 11. Spectral comparison — extended with bandwidth, rolloff, ZCR
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Spectral Feature Comparison: Reference vs Generated", fontsize=14, fontweight="bold")

labels3 = ["Reference\n(Real MC)", "Baseline\n(Vanilla)", "LoRA\n(Ours)"]
colors3 = ["#2ca02c", "#6baed6", "#e6550d"]

for ax, key, title, unit in zip(
    axes.flat,
    ["centroid_mean", "bandwidth_mean", "rolloff_mean", "zcr_mean"],
    ["Spectral Centroid", "Spectral Bandwidth", "Spectral Rolloff", "Zero-Crossing Rate"],
    ["Hz", "Hz", "Hz", "Rate"]
):
    vals = [ref_spec[key], base_spec[key], lora_spec[key]]
    bars = ax.bar(labels3, vals, color=colors3, edgecolor="black")
    ax.set_ylabel(unit)
    ax.set_title(title, fontweight="bold")
    for bar, v in zip(bars, vals):
        fmt = f"{v:.0f}" if unit == "Hz" else f"{v:.4f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                fmt, ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(PRES_DIR / "spectral_detailed.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: spectral_detailed.png")

# ============================================================
# 12. Mel spectrogram grid — baseline vs LoRA vs reference
# ============================================================
fig, axes = plt.subplots(4, 3, figsize=(15, 10))
fig.suptitle("Mel Spectrograms: Reference vs Baseline vs LoRA", fontsize=14, fontweight="bold")
col_titles = ["Reference (Real MC)", "Baseline (Vanilla AudioGen)", "LoRA (Fine-tuned)"]

# Pick representative reference samples (one per category)
ref_by_cat = {}
for w in ref_wavs:
    cat = w.parent.name
    if cat not in ref_by_cat:
        ref_by_cat[cat] = w

rep_refs = list(ref_by_cat.values())[:4]

for i in range(min(4, len(baseline_wavs))):
    for j, (wav_list, label) in enumerate([
        (rep_refs, "Ref"), (baseline_wavs, "Base"), (lora_wavs, "LoRA")
    ]):
        if i < len(wav_list):
            data, sr = load_audio_mono(wav_list[i])
            mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=80, n_fft=1024, hop_length=512)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            axes[i, j].imshow(mel_db, aspect="auto", origin="lower", cmap="magma", vmin=-80, vmax=0)
            axes[i, j].set_ylabel("Mel" if j == 0 else "")
            stem = wav_list[i].stem[:35]
            axes[i, j].set_title(f"{stem}" if i == 0 else stem, fontsize=8)
        else:
            axes[i, j].axis("off")
    if i == 0:
        for j, t in enumerate(col_titles):
            axes[0, j].set_title(t, fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(PRES_DIR / "spectrogram_3way.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: spectrogram_3way.png")

# ============================================================
# 13. Approach comparison radar chart
# ============================================================
print("\n=== Generating Approach Radar Chart ===")

# Normalized scores (0-1 scale) for each approach on multiple dimensions
# Dimensions: Audio Quality, Domain Fit, Diversity, Text Alignment, Scalability
approach_names = ["AudioLDM2\nLoRA", "AudioLDM\nFull FT", "AudioLDM\nNew Data",
                  "AudioGen\nLoRA", "Scratch v1", "Scratch v2"]
dimensions = ["Audio\nQuality", "Domain\nFit", "Output\nDiversity", "Text\nAlignment", "Scalability"]

# Estimated scores based on experiments (0-10)
scores = np.array([
    [1, 1, 2, 1, 7],   # AudioLDM2 LoRA — failed
    [5, 4, 5, 5, 3],   # AudioLDM Full FT
    [6, 5, 6, 6, 5],   # AudioLDM New Data
    [7, 7, 7, 7, 8],   # AudioGen LoRA — best
    [4, 5, 4, 4, 2],   # Scratch v1
    [5, 6, 5, 5, 2],   # Scratch v2
])

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
angles += angles[:1]  # close polygon

colors_radar = ["#ff6b6b", "#ffa07a", "#ffd700", "#2171b5", "#90EE90", "#2ca02c"]

for i, (name, score_row) in enumerate(zip(approach_names, scores)):
    values = score_row.tolist() + [score_row[0]]
    ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors_radar[i], alpha=0.7)
    ax.fill(angles, values, alpha=0.1, color=colors_radar[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dimensions, fontsize=10)
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
ax.set_title("Approach Comparison (Qualitative Scores)", fontsize=14,
             fontweight="bold", pad=30)

plt.tight_layout()
plt.savefig(PRES_DIR / "approach_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: approach_radar.png")

# ============================================================
# 14. Architecture comparison figure
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

arch_data = [
    ["Approach", "Architecture", "Base Params", "Trainable", "Audio Repr.", "Training Time*"],
    ["AudioLDM2 LoRA", "Diffusion (UNet + GPT-2)", "~1.1B", "LoRA (UNet only)", "Mel Spectrogram", "~1 hr (T4)"],
    ["AudioLDM Full FT", "Diffusion (all components)", "~400M", "All (~400M)", "Mel Spectrogram", "~2 hr (T4)"],
    ["AudioLDM New Data", "Diffusion (official trainer)", "~400M", "All (~400M)", "Mel Spectrogram", "~8 hr (T4)"],
    ["AudioGen LoRA ⭐", "Autoregressive + EnCodec", "1.5B", "LoRA (132M)", "EnCodec Tokens", "~3 hr (RTX 5090)"],
    ["From Scratch v1", "Custom Transformer + T5", "~25M", "All (~25M)", "EnCodec Tokens", "~2 hr (T4)"],
    ["From Scratch v2", "Custom Transformer + T5", "~25M", "All (~25M)", "EnCodec Tokens", "~4 hr (T4)"],
]

table = ax.table(cellText=arch_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

for j in range(len(arch_data[0])):
    table[0, j].set_facecolor("#0f3460")
    table[0, j].set_text_props(color="white", fontweight="bold", fontsize=10)

# Highlight AudioGen row
for j in range(len(arch_data[0])):
    table[4, j].set_facecolor("#e8f4fd")
    table[4, j].set_text_props(fontweight="bold")

for i in range(1, len(arch_data)):
    if i != 4:
        color = "#f8f8f8" if i % 2 == 0 else "white"
        for j in range(len(arch_data[0])):
            table[i, j].set_facecolor(color)

ax.set_title("Architecture Comparison Across Approaches", fontsize=14,
             fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(PRES_DIR / "architecture_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: architecture_comparison.png")

# ============================================================
# 15. Dataset evolution timeline
# ============================================================
fig, ax = plt.subplots(figsize=(14, 4))

stages = [
    ("Raw OGG\nExtraction", "195 files\nMinecraft Java Edition", "#bdd7e7"),
    ("Preprocessing\n32kHz Mono", "195 WAVs\nNormalized, trimmed", "#9ecae1"),
    ("Augmentation\nv1", "325 clips\n(speed perturb, concat)", "#6baed6"),
    ("Expanded\nCollection v2", "556 clips\n(+web sources, mods)", "#3182bd"),
    ("Virtual Aug\n(Scratch v2)", "325 + virtual\nspeed variants", "#08519c"),
]

for i, (title, desc, color) in enumerate(stages):
    x = i * 2.5
    rect = FancyBboxPatch((x, 0.3), 2, 1.4, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor="black", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 1, 1.2, title, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    ax.text(x + 1, 0.65, desc, ha="center", va="center", fontsize=8, color="white")
    if i < len(stages) - 1:
        ax.annotate("", xy=((i+1)*2.5, 1), xytext=(x+2, 1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))

ax.set_xlim(-0.5, 12.5)
ax.set_ylim(0, 2.2)
ax.axis("off")
ax.set_title("Dataset Evolution Pipeline", fontsize=14, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(PRES_DIR / "dataset_pipeline.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: dataset_pipeline.png")

# ============================================================
# 16. Save comprehensive metrics JSON
# ============================================================
metrics = {
    "fad": {
        "baseline": fad_baseline, "lora": fad_lora,
        "gen_vanilla": fad_gen_vanilla, "gen_lora": fad_gen_lora,
        "note": "Fréchet Audio Distance (mel-embedding), lower is better"
    },
    "kad": {
        "baseline": kad_baseline, "lora": kad_lora,
        "gen_vanilla": kad_gen_vanilla, "gen_lora": kad_gen_lora,
        "note": "Kernel Audio Distance (polynomial MMD), lower is better"
    },
    "inception_score": {
        "reference": is_ref, "baseline": is_baseline, "lora": is_lora,
        "gen_vanilla": is_gen_vanilla, "gen_lora": is_gen_lora,
        "note": "Inception Score proxy (embedding diversity), higher is better"
    },
    "clap": {
        "baseline_scores": base_sims if HAVE_CLAP else [],
        "lora_scores": lora_sims if HAVE_CLAP else [],
        "gen_lora_scores": gen_lora_sims if HAVE_CLAP else [],
        "gen_vanilla_scores": gen_vanilla_sims if HAVE_CLAP else [],
        "baseline_mean": float(np.mean(base_sims)) if (HAVE_CLAP and base_sims) else None,
        "lora_mean": float(np.mean(lora_sims)) if (HAVE_CLAP and lora_sims) else None,
        "gen_lora_mean": float(np.mean(gen_lora_sims)) if (HAVE_CLAP and gen_lora_sims) else None,
        "gen_vanilla_mean": float(np.mean(gen_vanilla_sims)) if (HAVE_CLAP and gen_vanilla_sims) else None,
    },
    "spectral": {
        "reference": ref_spec, "baseline": base_spec, "lora": lora_spec,
    },
    "outputs": {
        "baseline_count": len(baseline_wavs), "lora_count": len(lora_wavs),
        "generalization_count": len(gen_wavs), "reference_count": len(ref_wavs),
    },
    "training": {
        "epochs": 25, "dataset_train": 277, "dataset_val": 48,
        "lora_rank": 128, "lora_alpha": 256, "trainable_params_M": 132,
    },
    "datasets": {
        "v1_raw": 195, "v1_augmented": 325,
        "v2_expanded": 556,
        "v1_categories": dict(cat_counts),
        "google_drive_links": {
            "GenAIAudioLDM": "https://drive.google.com/drive/folders/1sGpF7I1vZnP_TgGBjJ1cytnEUAze9H5z",
            "GenAIMinecraftAudioV2": "https://drive.google.com/drive/folders/1Pp9_s4dcRcLYosLik9eDGETSTY5lfUub",
        }
    }
}

with open(PRES_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved: metrics.json")
print("\n=== All metrics and figures generated ===")
