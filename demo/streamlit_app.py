#!/usr/bin/env python3
"""
Minecraft AudioGen — Live Demo (Streamlit)
==========================================
Showcase: vanilla AudioGen baseline vs our LoRA fine-tuned model
side by side, with audio playback and spectrograms.

Two modes:
  • Showcase  — instant playback of pre-generated curated samples
  • Live      — generate on demand from any prompt (slower, requires GPU)

Run:
    & ".venv\\Scripts\\python.exe" -m streamlit run demo/streamlit_app.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import streamlit as st

# Make project src importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Minecraft AudioGen Demo",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & curated showcase samples
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_DIR = ROOT / "outputs" / "audiogen" / "baseline"
LORA_DIR = ROOT / "outputs" / "audiogen" / "lora"
GEN_DIR = ROOT / "outputs" / "audiogen" / "generalization"
LORA_WEIGHTS = ROOT / "outputs" / "audiogen" / "lora_weights" / "best"

# In-domain showcase: (display name, prompt, baseline_file, lora_file)
INDOMAIN_SAMPLES = [
    (
        "🦴 Skeleton death",
        "minecraft skeleton death sound effect",
        "minecraft_skeleton_death_sound_effect_001.wav",
        "minecraft_skeleton_death_sound_effect_001.wav",
    ),
    (
        "🧟 Zombie hurt",
        "minecraft zombie getting hurt sound effect",
        "minecraft_zombie_getting_hurt_sound_effect_000.wav",
        "minecraft_zombie_getting_hurt_sound_effect_000.wav",
    ),
    (
        "🕳️ Cave ambience",
        "minecraft cave ambience sound effect",
        "minecraft_cave_ambience_sound_effect_001.wav",
        "minecraft_cave_ambience_sound_effect_001.wav",
    ),
    (
        "👣 Footsteps on stone",
        "minecraft walking footsteps on stone surface sound effect",
        "minecraft_walking_footsteps_on_stone_surface_sound_effect_000.wav",
        "minecraft_walking_footsteps_on_stone_surface_sound_effect_000.wav",
    ),
]

# Generalization showcase: (display name, prompt, baseline_file, lora_file)
GENERALIZATION_SAMPLES = [
    (
        "🧟 Zombie groan in cave (4s)",
        "minecraft zombie groaning in a dark cave",
        "minecraft_zombie_groaning_in_a_dark_cave_4s_vanilla.wav",
        "minecraft_zombie_groaning_in_a_dark_cave_4s_lora.wav",
    ),
    (
        "🧟 Zombie groan in cave (8s)",
        "minecraft zombie groaning in a dark cave",
        "minecraft_zombie_groaning_in_a_dark_cave_8s_vanilla.wav",
        "minecraft_zombie_groaning_in_a_dark_cave_8s_lora.wav",
    ),
    (
        "🧟 Zombie groan in cave (12s)",
        "minecraft zombie groaning in a dark cave",
        "minecraft_zombie_groaning_in_a_dark_cave_12s_vanilla.wav",
        "minecraft_zombie_groaning_in_a_dark_cave_12s_lora.wav",
    ),
    (
        "💥 Creeper hiss + explosion + hurt (8s)",
        "creeper hiss then explosion, player hurt sound",
        "creeper_hiss_then_explosion_player_hurt__8s_vanilla.wav",
        "creeper_hiss_then_explosion_player_hurt__8s_lora.wav",
    ),
    (
        "💧 Cave + water + distant mobs (10s)",
        "cave ambience with water dripping and distant mobs",
        "cave_ambience_with_water_dripping_and_di_10s_vanilla.wav",
        "cave_ambience_with_water_dripping_and_di_10s_lora.wav",
    ),
    (
        "🔥 Blaze fireball whoosh impact (10s)",
        "blaze fireball whoosh impact explosion",
        "blaze_fireball_whoosh_impact_explosion_p_10s_vanilla.wav",
        "blaze_fireball_whoosh_impact_explosion_p_10s_lora.wav",
    ),
    (
        "👻 Skeleton hurt + ghast moan + damage (4s)",
        "skeleton hurt, ghast moan sound, player take damage",
        "skeleton_hurt_ghast_moan_sound_player_ta_4s_vanilla.wav",
        "skeleton_hurt_ghast_moan_sound_player_ta_4s_lora.wav",
    ),
    (
        "👻 Skeleton hurt + ghast moan + damage (8s)",
        "skeleton hurt, ghast moan sound, player take damage",
        "skeleton_hurt_ghast_moan_sound_player_ta_8s_vanilla.wav",
        "skeleton_hurt_ghast_moan_sound_player_ta_8s_lora.wav",
    ),
    (
        "👻 Skeleton hurt + ghast moan + damage (12s)",
        "skeleton hurt, ghast moan sound, player take damage",
        "skeleton_hurt_ghast_moan_sound_player_ta_12s_vanilla.wav",
        "skeleton_hurt_ghast_moan_sound_player_ta_12s_lora.wav",
    ),
    (
        "🏹 Footsteps then skeleton arrow (8s)",
        "footsteps on stone then skeleton arrow shot",
        "footsteps_on_stone_then_skeleton_arrow_s_8s_vanilla.wav",
        "footsteps_on_stone_then_skeleton_arrow_s_8s_lora.wav",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_audio(path_str: str) -> tuple[np.ndarray, int]:
    """Cached audio loading."""
    data, sr = sf.read(path_str, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


@st.cache_data(show_spinner=False)
def make_spectrogram_png(path_str: str, title: str = "") -> bytes:
    """Generate a log-power spectrogram PNG (scipy-only, numba-free)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal as sps

    y, sr = load_audio(path_str)

    # STFT via scipy — no numba required
    f, t, Sxx = sps.spectrogram(
        y, fs=sr, nperseg=1024, noverlap=768,
        scaling="spectrum", mode="magnitude",
    )
    log_S = 20.0 * np.log10(np.maximum(Sxx, 1e-8))
    log_S = np.clip(log_S - log_S.max(), -80.0, 0.0)

    fig, ax = plt.subplots(figsize=(5, 2.2), dpi=110)
    ax.pcolormesh(t, f / 1000.0, log_S, shading="auto", cmap="magma",
                  vmin=-80, vmax=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("kHz", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_audio_pair(prompt: str, baseline_path: Path, lora_path: Path) -> None:
    """Render a 2-column comparison of baseline vs LoRA audio."""
    st.markdown(f"##### Prompt: *{prompt}*")
    col_b, col_l = st.columns(2)

    with col_b:
        st.markdown("**🟦 Baseline AudioGen** (vanilla)")
        if baseline_path.exists():
            st.audio(str(baseline_path))
            st.image(make_spectrogram_png(str(baseline_path), "Baseline spectrogram"))
        else:
            st.warning(f"Missing: {baseline_path.name}")

    with col_l:
        st.markdown("**🟪 LoRA Fine-Tuned** (ours)")
        if lora_path.exists():
            st.audio(str(lora_path))
            st.image(make_spectrogram_png(str(lora_path), "LoRA spectrogram"))
        else:
            st.warning(f"Missing: {lora_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Live generation (lazy import to avoid loading torch on app startup)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AudioGen model (one-time)…")
def get_models():
    """Load both vanilla and LoRA AudioGen models. Cached across reruns."""
    import torch
    from audiocraft.models import AudioGen
    from src.mcaudio.train.audiogen_lora_train import inject_lora, load_lora_weights

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Baseline model
    baseline = AudioGen.get_pretrained("facebook/audiogen-medium")
    baseline.lm.float()
    baseline.compression_model.float()

    # LoRA model — separate copy with adapters loaded
    lora_model = AudioGen.get_pretrained("facebook/audiogen-medium")
    lora_model.lm.float()
    lora_model.compression_model.float()
    inject_lora(lora_model.lm, ["out_proj", "linear1", "linear2"],
                rank=128, alpha=256, dropout=0.05)
    load_lora_weights(lora_model.lm, LORA_WEIGHTS)
    lora_model.lm.to(device).float().eval()

    return baseline, lora_model, device


def live_generate(prompt: str, duration: float, temperature: float,
                  cfg_coef: float, seed: int) -> tuple[Path, Path]:
    """Generate baseline + LoRA samples for a prompt. Returns (baseline_path, lora_path)."""
    import torch

    baseline, lora_model, _ = get_models()

    out_dir = ROOT / "outputs" / "audiogen" / "demo_live"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe = "".join(c if c.isalnum() else "_" for c in prompt)[:40]
    ts = int(time.time())

    paths: dict[str, Path] = {}
    for name, model in (("baseline", baseline), ("lora", lora_model)):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.set_generation_params(
            duration=duration, use_sampling=True, top_k=250,
            top_p=0.0, temperature=temperature, cfg_coef=cfg_coef,
        )
        with torch.no_grad():
            wav = model.generate([prompt])
        audio = wav[0].cpu().squeeze().numpy()
        fname = out_dir / f"{safe}_{ts}_{name}.wav"
        sf.write(str(fname), audio, samplerate=model.sample_rate, subtype="FLOAT")
        paths[name] = fname

    return paths["baseline"], paths["lora"]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎮 Minecraft AudioGen")
    st.caption("Domain-specialized text-to-audio · CSCI 595")

    mode = st.radio(
        "Demo mode",
        ["Showcase (instant)", "Live generation"],
        index=0,
        help="Showcase plays pre-generated curated samples. Live runs the model on demand (slower).",
    )

    st.markdown("---")
    st.markdown("### Model")
    st.markdown(
        "- **Base:** `facebook/audiogen-medium` (1.5B)\n"
        "- **LoRA:** rank=128, α=256\n"
        "- **Trainable:** ~132M params\n"
        "- **Train:** 277 clips, 25 epochs, FP32"
    )

    st.markdown("---")
    st.markdown("### Metrics (LoRA vs Baseline)")
    st.markdown(
        "| | Baseline | LoRA | Δ |\n"
        "|---|---:|---:|---:|\n"
        "| FAD ↓ | 6171 | **3856** | **−37%** |\n"
        "| KAD ↓ | 9.1e8 | **2.5e8** | **−73%** |\n"
        "| CLAP ↑ | 0.142 | **0.162** | +14% |"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎮 Minecraft Sound Effects — AudioGen LoRA Demo")
st.markdown(
    "Compare **vanilla AudioGen** against our **LoRA fine-tuned** model "
    "specialized for Minecraft-style sound effects."
)

if mode == "Showcase (instant)":
    tab_in, tab_gen = st.tabs(["🎯 In-Domain Prompts", "🚀 Generalization Prompts"])

    with tab_in:
        st.markdown(
            "Standard Minecraft sound categories the model was trained on. "
            "Both models received the same prompt & seed."
        )
        names = [s[0] for s in INDOMAIN_SAMPLES]
        sel = st.selectbox("Pick a sample:", names, key="indomain_sel")
        idx = names.index(sel)
        _, prompt, base_f, lora_f = INDOMAIN_SAMPLES[idx]
        render_audio_pair(prompt, BASELINE_DIR / base_f, LORA_DIR / lora_f)

    with tab_gen:
        st.markdown(
            "Unseen multi-event prompts and longer durations (4s–12s) to test "
            "generalization beyond the training set."
        )
        names = [s[0] for s in GENERALIZATION_SAMPLES]
        sel = st.selectbox("Pick a sample:", names, key="gen_sel")
        idx = names.index(sel)
        _, prompt, base_f, lora_f = GENERALIZATION_SAMPLES[idx]
        render_audio_pair(prompt, GEN_DIR / base_f, GEN_DIR / lora_f)

else:
    st.markdown(
        "⚠️ **Live mode** loads both models into VRAM (~6 GB). "
        "First run takes ~30s; each generation takes 10–30s on RTX 5090."
    )

    presets = [
        "minecraft zombie groaning in a dark cave",
        "minecraft creeper hiss then explosion",
        "minecraft skeleton shooting arrow",
        "minecraft cave ambience with water dripping",
        "minecraft enderman teleport sound",
        "minecraft footsteps on grass then jump",
        "minecraft ghast moaning fireball whoosh",
        "minecraft blaze fireball impact explosion",
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_input(
            "Prompt", value=presets[0],
            help="Try descriptive Minecraft prompts. Tip: include 'minecraft' for in-domain bias."
        )
    with col2:
        preset = st.selectbox("Or pick a preset:", [""] + presets, index=0)
        if preset:
            prompt = preset

    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
    with cfg_col1:
        duration = st.slider("Duration (s)", 2.0, 12.0, 4.0, step=1.0)
    with cfg_col2:
        temperature = st.slider("Temperature", 0.5, 1.5, 0.9, step=0.05)
    with cfg_col3:
        cfg_coef = st.slider("CFG coef", 1.0, 6.0, 3.0, step=0.5)
    with cfg_col4:
        seed = st.number_input("Seed", min_value=0, value=42, step=1)

    if st.button("🎵 Generate (baseline + LoRA)", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a prompt.")
        else:
            with st.spinner("Generating both samples — this can take 20–60s…"):
                t0 = time.time()
                baseline_path, lora_path = live_generate(
                    prompt.strip(), duration, temperature, cfg_coef, int(seed)
                )
                elapsed = time.time() - t0
            st.success(f"Generated in {elapsed:.1f}s")
            render_audio_pair(prompt, baseline_path, lora_path)


st.markdown("---")
st.caption(
    "🎓 CSCI 595 Generative AI · Final Project · "
    "[GenAI-Minecraft-Sounds](https://github.com/BHatiru/GenAI-Minecraft-Sounds)"
)
