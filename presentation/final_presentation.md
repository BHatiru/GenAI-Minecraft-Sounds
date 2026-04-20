---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #fafafa;
  }
  section.lead {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
  }
  section.lead h1 {
    font-size: 2.2em;
    color: #e8e8e8;
  }
  section.lead h2 {
    font-size: 1.2em;
    font-weight: 400;
    color: #aaa;
  }
  section.section-header {
    background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
    color: white;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.section-header h1 {
    font-size: 2.5em;
  }
  section.section-header h2 {
    font-weight: 400;
    color: #ccc;
  }
  table { font-size: 0.72em; margin: 0 auto; }
  th { background: #0f3460; color: white; padding: 4px 10px; }
  td { padding: 3px 10px; }
  code { font-size: 0.78em; background: #e8e8e8; padding: 2px 6px; border-radius: 3px; }
  img[alt~="center"] {
    display: block;
    margin: 0 auto;
  }
  h1 { color: #0f3460; font-size: 1.6em; }
  h2 { color: #16213e; font-size: 1.2em; }
  h3 { color: #533483; font-size: 1.0em; }
  .columns { display: flex; gap: 20px; }
  .col { flex: 1; }
  strong { color: #0f3460; }
  blockquote { border-left: 4px solid #533483; background: #f0e6ff; padding: 6px 12px; font-size: 0.85em; }
  footer { font-size: 0.6em; color: #888; }
  ul, ol { font-size: 0.9em; margin: 0.3em 0; }
  li { margin: 0.15em 0; }
  p { margin: 0.4em 0; }
---

<!-- _class: lead -->

# 🎮 Domain-Specialized Text-to-Audio Generation
# for Minecraft Sound Effects

## CSCI 595 — Generative AI · Final Presentation

**Team:** Batyr Bodaubay · Team Members
**Date:** April 2026

---

<!-- _class: section-header -->

# 1. Introduction
## Problem Definition & Motivation

---

# Problem Statement

### Can we generate **Minecraft-style sound effects** from text descriptions?

<div class="columns">
<div class="col">

**Why Minecraft Audio?**
- Iconic 8-bit retro game aesthetic
- Distinctive low-fidelity, blocky sound design
- Rich taxonomy: mobs, ambient, footsteps, combat, damage
- No existing text-to-audio model captures this domain

</div>
<div class="col">

**Goal**
- Text prompt → Minecraft-style audio
  - *"minecraft zombie groaning in a dark cave"*
  - *"creeper hiss then explosion"*
  - *"footsteps on stone blocks"*
- Explore **multiple approaches**: fine-tuning pretrained models vs. training from scratch

</div>
</div>

---

# Dataset Overview

<div class="columns">
<div class="col">

**Raw:** 195 `.ogg` from Minecraft Java Edition

**Augmented:** 325 clips (32 kHz mono WAV)
- Speed perturbation: 0.85×, 1.15×
- Concatenation for sequences

**Extended:** 556 WAVs (AudioLDM)

</div>
<div class="col">

| Category | Raw | Aug |
|---|---:|---:|
| Hostile Mobs | 74 | 148 |
| Ambient/Cave | 50 | 80 |
| Footsteps | 58 | 72 |
| Damage/Combat | 13 | 25 |
| **Total** | **195** | **325** |

</div>
</div>

---

# Dataset Evolution Pipeline

![center w:950](dataset_pipeline.png)

---

# Dataset Composition: v1 vs v2

![center w:900](dataset_composition_both.png)

| Dataset | Clips | Usage |
|---|---:|---|
| **v1** | 325 | AudioGen LoRA, From Scratch |
| **v2** | 556 | AudioLDM Full FT, New Data |

---

# Dataset Splits Across Approaches

![center w:850](dataset_splits_comparison.png)

---

<!-- _class: section-header -->

# 2. Challenges
## Technical Obstacles Encountered

---

# Key Challenges

<div class="columns">
<div class="col">

### Data & Model
- **Tiny dataset** — 195 raw clips (<15 min)
- **Class imbalance** — mobs overrepresented
- **AudioLDM2 LoRA → noise** — failed
- **FP16 instability** — AudioCraft NaN

</div>
<div class="col">

### Infrastructure
- **VRAM limits** — Colab T4 too small
- **Device mismatches** — LoRA CPU vs GPU
- **API drift** — torchaudio changes

### Pivoting
- Initial AudioLDM2 LoRA → **failure**
- Explored 6 approaches in parallel
- Converged on **AudioGen LoRA**

</div>
</div>

---

<!-- _class: section-header -->

# 3. Related Works
## Foundation Models & Techniques

---

# Related Works

<div class="columns">
<div class="col">

### Text-to-Audio Models
- **AudioLDM** (2023) — Latent diffusion + CLAP
- **AudioLDM2** (2024) — GPT-2 + diffusion
- **AudioGen** (2023) — Autoregressive EnCodec
- **MusicGen** (2023) — Music-focused variant

### Key Techniques
- **LoRA** — Low-rank adaptation
- **EnCodec** — Neural audio codec
- **CLAP** — Contrastive audio-text

</div>
<div class="col">

### Our Contribution
- Systematic comparison of **6 approaches**
- Focus on **retro game SFX** domain
- First work on Minecraft audio generation

### Evaluation
- **FAD** — Distributional distance
- **KAD** — MMD-based metric
- **CLAP** — Text-audio alignment
- **Spectral** — Centroid, bandwidth

</div>
</div>

---

<!-- _class: section-header -->

# 4. Methodology & Results
## Six Approaches Explored

---

# Approach Overview

| # | Approach | Model | Architecture | Training | Status |
|---|---|---|---|---|---|
| 1 | AudioLDM2 LoRA | `cvssp/audioldm2` | Diffusion + UNet | 100–500 steps | ❌ Failed |
| 2 | AudioLDM Full FT | `cvssp/audioldm` | Diffusion (all components) | 3 epochs | ✅ Complete |
| 3 | AudioLDM New Data | `audioldm-s-full` | Diffusion (official trainer) | 50 epochs (17K steps) | ✅ Complete |
| 4 | **AudioGen LoRA** | `facebook/audiogen-medium` | **Autoregressive + EnCodec** | **25 epochs** | ✅ **Primary** |
| 5 | From Scratch (v1) | Custom Transformer + T5 | Causal decoder + cross-attn | 50 epochs | ✅ Complete |
| 6 | From Scratch (v2) | Custom Transformer + T5 | Same + virtual augmentation | 100 epochs | ✅ Complete |

---

# Approach 1: AudioLDM2 LoRA (Initial Attempt)

<div class="columns">
<div class="col">

### Configuration
- **Model:** `cvssp/audioldm2`
- **LoRA targets:** UNet cross-attention (`to_k`, `to_q`, `to_v`, `to_out.0`)
- **Rank:** 8, Alpha: 16
- **Steps:** 100–500
- **Hardware:** Colab T4 (15 GB)
- **Precision:** FP16 mixed

### Why It Failed
- Output was **pure noise/artifacts**
- AudioLDM2's complex architecture (GPT-2 + diffusion) resists LoRA on UNet alone
- Full UNet fine-tuning also produced noise
- Insufficient training data for diffusion model convergence

</div>
<div class="col">

### Lesson Learned

> "LoRA and last-layer fine-tuning of AudioLDM2 produce poor results on small niche datasets. The multi-stage architecture requires coordinated adaptation across all components."

**Impact on Project:**
- Led to systematic exploration of 5 alternative approaches
- Each team member pursued a different strategy
- Documented in `docs/approaches.md`

</div>
</div>

---

# Approach 2: AudioLDM Full Fine-Tuning

<div class="columns">
<div class="col">

### Configuration
| Parameter | Value |
|---|---|
| Model | `cvssp/audioldm` |
| Components | All (UNet, VAE, vocoder) |
| Epochs | 3 |
| LR | UNet: 1e-6, others: 5e-7 |
| Precision | FP32 |

### Multi-Component Loss
- Diffusion MSE: **1.0**
- VAE recon (L1): **0.5**
- Vocoder/Waveform: **0.1** each

</div>
<div class="col">

### Results
- 3 epochs completed
- Multi-loss tracking works
- FP32 stable but slow

### Key Insight
> Multi-component loss keeps all pipeline stages coherent. 3 epochs insufficient for full domain adaptation.

</div>
</div>

---

# Approach 3: AudioLDM with Expanded Dataset

<div class="columns">
<div class="col">

### Configuration
| Parameter | Value |
|---|---|
| Model | `audioldm-s-full` |
| Dataset | **556 WAVs** |
| Epochs | 50 (17K steps) |
| LR | 5e-5 |

### CLAP Analysis
- Intra-class cosine similarity improved
- t-SNE shows better category clustering

</div>
<div class="col">

### Results
- Training converged over 17K steps
- CLAP text-audio alignment improved

### Key Insight
> Larger dataset (556 vs 325) + more epochs improves diffusion convergence

</div>
</div>

---

# Approach 4: AudioGen LoRA ⭐ (Primary)

<div class="columns">
<div class="col">

### Why AudioGen?
- **Autoregressive** on EnCodec tokens
- `facebook/audiogen-medium` — **1.5B params**
- 4 codebooks × 2048 vocab × 50Hz

### LoRA Config
| Param | Value |
|---|---|
| Rank | 128 |
| Alpha | 256 |
| Targets | `out_proj`, `linear1/2` |
| Trainable | **132M** |

</div>
<div class="col">

### Training Config
| Param | Value |
|---|---|
| Dataset | 277 train / 48 val |
| Epochs | **25** |
| LR | 1e-4, cosine |
| Precision | **FP32** |
| Hardware | **RTX 5090** |

### Critical Details
- LoRA injected **before** `.to(device)`
- FP32 required (AudioCraft NaN bug)

</div>
</div>

---

# AudioGen LoRA — Results

<div class="columns">
<div class="col">

### Training
- 25 epochs completed
- 7 checkpoints saved (~504 MB each)

### Samples Generated
| Type | Count |
|---|---:|
| Baseline | 8 |
| LoRA | 8 |
| Generalization | 22 |
| **Total** | **38** |

</div>
<div class="col">

### Spectrogram: Baseline vs LoRA

![center w:450](spectrograms_comparison.png)

LoRA shows **denser, more structured** patterns

</div>
</div>

---

# AudioGen LoRA — CLAP & Spectral

<div class="columns">
<div class="col">

### CLAP Similarity
![center w:420](clap_similarity.png)

LoRA (0.162) ≈ Baseline (0.152)

</div>
<div class="col">

### Spectral Features
![center w:420](spectral_comparison.png)

LoRA centroid: 1603 Hz vs Ref: 664 Hz

</div>
</div>

---

# 3-Way Spectrogram: Reference vs Baseline vs LoRA

![center w:800](spectrogram_3way.png)

LoRA produces spectral patterns closer to real Minecraft audio

---

# Evaluation Metrics: FAD / KAD / IS

![center w:900](fad_kad_is_comparison.png)

| Metric | Baseline | LoRA | Improvement |
|---|---:|---:|---:|
| FAD ↓ | 6171 | 3856 | **37%** |
| KAD ↓ | 9.1e8 | 2.5e8 | **73%** |

---

# CLAP Text-Audio Similarity

![center w:850](clap_detailed.png)

| Config | Mean | Std |
|---|---:|---:|
| Baseline | 0.142 | ±0.21 |
| LoRA | 0.162 | ±0.16 |
| Gen-LoRA | 0.270 | ±0.12 |

---

# Spectral Feature Analysis

![center w:750](spectral_detailed.png)

LoRA bandwidth (1456 Hz) and ZCR (0.132) closer to reference than baseline

---

# Metrics Summary

![center w:900](metrics_table.png)

**Key:** LoRA improves FAD by 37%, KAD by 73% over baseline

---

# Approach 5: Train from Scratch (v1)

<div class="columns">
<div class="col">

### Architecture
- Custom causal transformer decoder
- Cross-attention to frozen **T5-small**
- EnCodec tokens (1.5 kbps, 2 codebooks)

| Param | Value |
|---|---|
| Layers | 10 |
| d_model | 512 |
| Heads | 8 |

</div>
<div class="col">

### Training
| Param | Value |
|---|---|
| Dataset | 325 (70/15/15) |
| Epochs | 50 |
| LR | 3e-4 |

### Results
- Converged over 50 epochs
- Quality limited by small dataset

</div>
</div>

---

# Approach 6: Train from Scratch (v2 + Virtual Aug)

<div class="columns">
<div class="col">

### Key Difference
- **All clips → train** (max data)
- Val/test use **virtual augmented** copies
  - Speed: 0.95×, 1.05×, 1.08×
- **100 epochs** (2× v1)

</div>
<div class="col">

### Results
- 100-epoch training completed
- Multiple inference rounds tested
- Continued improvement over v1

### Key Insight
> Virtual aug allows using **all real data** for training

</div>
</div>

---

<!-- _class: section-header -->

# 5. Comparative Analysis
## Cross-Approach Summary

---

# Cross-Approach Comparison

| Approach | Arch | Params | Quality |
|---|---|---|---|
| AudioLDM2 LoRA | Diffusion | 1.1B | ❌ Noise |
| AudioLDM Full FT | Diffusion | 400M | ⚠️ Partial |
| AudioLDM New Data | Diffusion | 400M | ⚠️ Conv. |
| **AudioGen LoRA** | **Autoreg** | **1.5B** | **✅ Best** |
| From Scratch v1 | Transformer | 25M | ⚠️ Limited |
| From Scratch v2 | Transformer | 25M | ⚠️ Better |

---

# Architecture Comparison

![center w:900](architecture_comparison.png)

---

# Approach Radar Chart

![center w:550](approach_radar.png)

**AudioGen LoRA** strongest overall; **From Scratch v2** best domain fit among small models

---

# Key Findings

<div class="columns">
<div class="col">

### What Worked
- **AudioGen + LoRA** — autoregressive handles small datasets
- **FP32 training** — essential for AudioCraft
- **LoRA rank 128** — sufficient capacity
- **Speed perturbation** — effective augmentation

</div>
<div class="col">

### What Didn't Work
- **AudioLDM2 LoRA** — complex arch resists partial adaptation
- **Small dataset + diffusion** — needs more data
- **FP16** — causes NaN in AudioCraft
- **3 epochs** — insufficient for domain shift

</div>
</div>

---

<!-- _class: section-header -->

# 6. Demonstration
## Audio Samples & Spectrograms

---

# Demo: Generalization Spectrograms

Unseen multi-event and combo prompts at various durations (4s–12s):

![center w:900](generalization_spectrograms.png)

---

# Demo: Sample Prompts

| Prompt | Dur | Type |
|---|---|---|
| *"minecraft cave ambience"* | 4s | In-domain |
| *"skeleton death sound"* | 4s | In-domain |
| *"creeper hiss then explosion"* | 8s | Multi-event |
| *"cave + water dripping + distant mobs"* | 10s | Layered |
| *"skeleton, ghast, player damage"* | 12s | Combo |

> 🎧 **38 WAV samples** available for live demo

---

<!-- _class: section-header -->

# 7. Conclusions & Future Work

---

# Conclusions

<div class="columns">
<div class="col">

### Summary
- Explored **6 approaches** for Minecraft sound generation
- **AudioGen LoRA** — most promising
  - FAD improved **37%**, KAD improved **73%**
  - Generalizes to unseen prompts
- **Train-from-scratch** viable but limited by data

</div>
<div class="col">

### Future Work
- **Larger dataset** — modded sound packs
- **Longer training** — 100+ epochs
- **Perceptual loss** — CLAP objective
- **Post-processing** — bit-crush for retro aesthetic
- **Human eval** — listening tests

</div>
</div>

---

<!-- _class: lead -->

# Thank You

### Questions?

**Repository:** github.com/BHatiru/GenAI-Minecraft-Sounds
**38 generated samples** available for listening
**7 LoRA checkpoints** saved for reproducibility
