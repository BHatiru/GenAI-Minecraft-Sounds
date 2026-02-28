#!/usr/bin/env python3
"""
lora_train.py
─────────────
LoRA fine-tuning of AudioLDM2's UNet on Minecraft sound effects.

Pipeline
────────
1. Load the full AudioLDM2 pipeline (VAE, UNet, text encoders, vocoder,
   tokenizers, feature extractor, scheduler).
2. Freeze everything, apply LoRA adapters to the UNet cross-attention
   projections.
3. For each training step:
     a. Load a batch of (waveform, caption) pairs.
     b. Convert waveforms → mel spectrograms → VAE latents.
     c. Encode captions through both text encoders + language model
        to get the prompt embeddings that the UNet expects.
     d. Sample random noise & timestep, add noise to latents.
     e. Predict noise with UNet (conditioned on prompt embeddings).
     f. MSE loss between predicted and actual noise.
4. Save LoRA adapter weights.

Designed for a single T4 GPU on Google Colab (≈15 GB VRAM).

Usage
─────
    python -m src.mcaudio.train.lora_train --config configs/demo1.yaml
    python -m src.mcaudio.train.lora_train --config configs/demo1.yaml --max_steps 200
"""
from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════

class McAudioDataset(Dataset):
    """
    Reads manifest.csv and yields (waveform, caption) tuples.
    Waveforms are already preprocessed to 16 kHz mono float32.
    """

    def __init__(self, manifest_csv: str, processed_dir: str, split: str = "train"):
        self.processed_dir = Path(processed_dir)
        self.items: list[dict[str, str]] = []

        with open(manifest_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.items.append(row)
        log.info("McAudioDataset: loaded %d items (split=%s)", len(self.items), split)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        row = self.items[idx]
        path = self.processed_dir / row["file_name"]
        audio, sr = sf.read(str(path), dtype="float32")
        return {"audio": torch.from_numpy(audio), "caption": row["caption"]}


# ═══════════════════════════════════════════════════════════════════
#  Audio → MEL → Latent helpers
# ═══════════════════════════════════════════════════════════════════

def waveform_to_mel(
    waveform: torch.Tensor,
    feature_extractor,
    target_length: int = 512,
) -> torch.Tensor:
    """
    Convert a batch of waveforms to mel spectrogram tensors
    suitable for the AudioLDM2 VAE.

    Parameters
    ----------
    waveform : (B, T) float32 tensor at 16 kHz
    feature_extractor : the pipeline's feature_extractor
    target_length : mel time-frames (512 for ≈4 s at 16 kHz)

    Returns
    -------
    mel : (B, 1, freq_bins, target_length) float32 tensor
    """
    # feature_extractor expects list of numpy arrays
    batch_np = [w.cpu().numpy() for w in waveform]
    features = feature_extractor(
        batch_np,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding="max_length",
        max_length=target_length,
        truncation=True,
    )
    # Shape: (B, freq_bins, time) → add channel dim → (B, 1, freq, time)
    mel = features["input_features"]
    if mel.dim() == 3:
        mel = mel.unsqueeze(1)
    return mel


@torch.no_grad()
def encode_audio_to_latents(
    waveform: torch.Tensor,
    vae,
    feature_extractor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Waveform → mel → VAE encoder → latent (B, C, H, W)."""
    mel = waveform_to_mel(waveform, feature_extractor)
    mel = mel.to(device=device, dtype=dtype)
    posterior = vae.encode(mel).latent_dist
    latents = posterior.sample() * vae.config.scaling_factor
    return latents


@torch.no_grad()
def encode_prompt(
    captions: list[str],
    pipe,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Encode text captions through AudioLDM2's dual-encoder pipeline.

    Returns
    -------
    prompt_embeds : (B, seq_len, dim)
    attention_mask : (B, seq_len)  or None
    generated_prompt_embeds : (B, gen_seq_len, dim)
    """
    result = pipe.encode_prompt(
        prompt=captions,
        device=device,
        num_waveforms_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    # encode_prompt may return 3 or more values
    prompt_embeds = result[0]
    attention_mask = result[1]
    generated_prompt_embeds = result[2]
    return (
        prompt_embeds.to(dtype=dtype),
        attention_mask,
        generated_prompt_embeds.to(dtype=dtype),
    )


# ═══════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════

def train(
    manifest_csv: str,
    processed_dir: str,
    model_id: str = "cvssp/audioldm2",
    output_dir: str = "outputs/demo1/lora_weights",
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_train_steps: int = 500,
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 50,
    mixed_precision: str = "fp16",
    seed: int = 42,
    log_every: int = 10,
    save_every: int = 100,
) -> None:
    """Train LoRA adapters on AudioLDM2's UNet."""
    from diffusers import AudioLDM2Pipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        weight_dtype = torch.float16 if mixed_precision == "fp16" else torch.float32
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        weight_dtype = torch.float32
    else:
        device = torch.device("cpu")
        weight_dtype = torch.float32

    log.info("Device: %s  |  weight dtype: %s", device, weight_dtype)

    # ── Load pipeline ────────────────────────────────────────────
    log.info("Loading AudioLDM2 pipeline: %s", model_id)
    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=weight_dtype)
    pipe = pipe.to(device)

    # Apply same compatibility patches as generate.py
    if hasattr(pipe, "text_encoder"):
        _orig_gtf = pipe.text_encoder.get_text_features
        def _patched_gtf(*args, **kwargs):
            out = _orig_gtf(*args, **kwargs)
            if hasattr(out, "pooler_output"):
                return out.pooler_output
            if isinstance(out, tuple):
                return out[1]
            return out
        pipe.text_encoder.get_text_features = _patched_gtf

    from transformers.models.gpt2.modeling_gpt2 import GPT2Model as _GPT2Model
    if isinstance(pipe.language_model, _GPT2Model):
        from transformers import GPT2LMHeadModel
        _orig_lm = pipe.language_model
        _lm_head = GPT2LMHeadModel(_orig_lm.config)
        _lm_head.transformer.load_state_dict(_orig_lm.state_dict())
        _lm_head = _lm_head.to(device=device, dtype=weight_dtype)
        _orig_fwd = _lm_head.forward
        def _patched_fwd(*args, **kwargs):
            kwargs["output_hidden_states"] = True
            out = _orig_fwd(*args, **kwargs)
            if not hasattr(out, "last_hidden_state") and hasattr(out, "hidden_states"):
                out.last_hidden_state = out.hidden_states[-1]
            return out
        _lm_head.forward = _patched_fwd
        pipe.language_model = _lm_head
        log.info("Swapped GPT2Model → GPT2LMHeadModel")

    unet = pipe.unet
    vae = pipe.vae
    feature_extractor = pipe.feature_extractor

    # ── Freeze everything ────────────────────────────────────────
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.language_model.requires_grad_(False)
    unet.requires_grad_(False)

    # ── Apply LoRA to UNet ───────────────────────────────────────
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # ── Noise scheduler for training ─────────────────────────────
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )

    # ── Dataset & dataloader ─────────────────────────────────────
    dataset = McAudioDataset(manifest_csv, processed_dir, split="train")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    log.info("Training set: %d samples, batch_size=%d, grad_accum=%d",
             len(dataset), batch_size, gradient_accumulation_steps)
    log.info("Effective batch size: %d", batch_size * gradient_accumulation_steps)

    # ── Optimizer & scheduler ────────────────────────────────────
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)

    total_steps = max_train_steps
    warmup_steps = min(warmup_steps, total_steps // 5)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        if lr_scheduler_type == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0  # constant

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Mixed precision scaler ───────────────────────────────────
    use_amp = (device.type == "cuda" and weight_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Training loop ────────────────────────────────────────────
    log.info("Starting LoRA training for %d steps …", max_train_steps)
    global_step = 0
    running_loss = 0.0
    unet.train()

    while global_step < max_train_steps:
        for batch in loader:
            if global_step >= max_train_steps:
                break

            waveforms = batch["audio"]           # (B, T)
            captions = batch["caption"]          # list[str]

            # ── Encode audio → latents (no grad) ─────────────────
            latents = encode_audio_to_latents(
                waveforms, vae, feature_extractor,
                device=device, dtype=weight_dtype,
            )

            # ── Encode text prompts (no grad) ────────────────────
            prompt_embeds, attn_mask, gen_prompt_embeds = encode_prompt(
                captions, pipe, device=device, dtype=weight_dtype,
            )

            # ── Forward diffusion: add noise ─────────────────────
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device, dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # ── Predict noise with UNet ──────────────────────────
            with torch.amp.autocast("cuda", enabled=use_amp):
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=gen_prompt_embeds,
                    encoder_hidden_states_1=prompt_embeds,
                    encoder_attention_mask_1=attn_mask,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(noise_pred, noise)
                loss = loss / gradient_accumulation_steps

            # ── Backward ─────────────────────────────────────────
            scaler.scale(loss).backward()

            if (global_step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            running_loss += loss.item() * gradient_accumulation_steps

            if global_step % log_every == 0:
                avg = running_loss / max(log_every, 1)
                log.info(
                    "  step %4d / %d  |  loss %.4f  |  lr %.2e",
                    global_step, max_train_steps, avg,
                    scheduler.get_last_lr()[0],
                )
                running_loss = 0.0

            if save_every and global_step > 0 and global_step % save_every == 0:
                ckpt_dir = Path(output_dir) / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                unet.save_pretrained(str(ckpt_dir))
                log.info("  ↳ saved checkpoint → %s", ckpt_dir)

            global_step += 1

    # ── Save final weights ───────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(out))
    log.info("LoRA weights saved to %s", out)
    log.info("Done – trained %d steps.", global_step)


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tune AudioLDM2 UNet on Minecraft sounds.")
    p.add_argument("--config", type=str, default="configs/demo1.yaml")
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--processed", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    lora_cfg = cfg.get("lora", {})

    kwargs = dict(
        manifest_csv=args.manifest or paths.get("manifest", "data/manifest.csv"),
        processed_dir=args.processed or paths.get("processed", "data/processed"),
        model_id=cfg.get("inference", {}).get("model_id", "cvssp/audioldm2"),
        output_dir=args.output or paths.get("lora_weights", "outputs/demo1/lora_weights"),
        rank=lora_cfg.get("rank", 8),
        alpha=lora_cfg.get("alpha", 16),
        dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg.get("target_modules", None),
        learning_rate=lora_cfg.get("learning_rate", 1e-4),
        batch_size=lora_cfg.get("batch_size", 1),
        gradient_accumulation_steps=lora_cfg.get("gradient_accumulation_steps", 4),
        max_train_steps=args.max_steps or lora_cfg.get("max_train_steps", 500),
        lr_scheduler_type=lora_cfg.get("lr_scheduler", "cosine"),
        warmup_steps=lora_cfg.get("warmup_steps", 50),
        mixed_precision=lora_cfg.get("mixed_precision", "fp16"),
        seed=lora_cfg.get("seed", 42),
        log_every=args.log_every if args.log_every is not None else 10,
        save_every=args.save_every if args.save_every is not None else 100,
    )

    train(**kwargs)


if __name__ == "__main__":
    main()
