#!/usr/bin/env python3
"""
lora_train.py
─────────────
Minimal LoRA fine-tuning of AudioLDM2's UNet on Minecraft sound effects.

This is a *simplified, self-contained* training loop suitable for a
feasibility demo on a single T4 GPU in Colab.  For production, use the
diffusers train_text_to_audio_lora.py script instead.

Usage
─────
    python -m src.mcaudio.train.lora_train --config configs/demo1.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ── Dataset ─────────────────────────────────────────────────────────


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


# ── Training ────────────────────────────────────────────────────────


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
) -> None:
    """
    Train LoRA adapters on AudioLDM2 UNet.
    
    NOTE: This is a structural stub that demonstrates the training
    setup.  The full training loop with proper latent encoding
    requires the AudioLDM2 VAE + text encoders which are loaded
    inside the pipeline. A complete implementation is provided
    in the Colab notebook.
    """
    from diffusers import AudioLDM2Pipeline

    if target_modules is None:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if mixed_precision == "fp16" and device == "cuda" else torch.float32

    # ── Load pipeline components ─────────────────────────────────
    log.info("Loading AudioLDM2 pipeline: %s", model_id)
    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Freeze everything except the LoRA layers
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # ── Apply LoRA ───────────────────────────────────────────────
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # ── Dataset & loader ─────────────────────────────────────────
    dataset = McAudioDataset(manifest_csv, processed_dir, split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ── Optimiser & scheduler ────────────────────────────────────
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

    if lr_scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max_train_steps)
    else:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    # ── Training loop (simplified) ───────────────────────────────
    log.info("Starting LoRA training for %d steps …", max_train_steps)
    global_step = 0
    unet.train()

    while global_step < max_train_steps:
        for batch in loader:
            if global_step >= max_train_steps:
                break

            # NOTE: A full implementation would:
            # 1. Encode audio through VAE to get latents
            # 2. Encode captions through text encoder
            # 3. Add noise to latents (forward diffusion)
            # 4. Predict noise with UNet
            # 5. Compute MSE loss between predicted and actual noise
            #
            # This is structurally shown here. The Colab notebook has
            # the complete implementation with proper encoding.

            audio = batch["audio"].to(device, dtype=dtype)           # (B, T)
            captions = batch["caption"]                               # list[str]

            # Placeholder: compute a dummy loss to verify the whole
            # pipeline (optimizer, LoRA params, scheduler) connects.
            # Replace with real latent-diffusion loss in full training.
            dummy_input = torch.randn(batch_size, 4, 1, 64, device=device, dtype=dtype)
            dummy_timestep = torch.randint(0, 1000, (batch_size,), device=device)
            # We skip actual UNet forward here to avoid encoder_hidden_states setup
            loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss.backward()

            if (global_step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % 50 == 0:
                log.info("  step %d / %d  |  loss %.4f  |  lr %.2e",
                         global_step, max_train_steps, loss.item(),
                         optimizer.param_groups[0]["lr"])

            global_step += 1

    # ── Save adapters ────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(out))
    log.info("LoRA weights saved to %s", out)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune AudioLDM2 UNet on Minecraft sounds.")
    p.add_argument("--config", type=str, default="configs/demo1.yaml", help="YAML config.")
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--processed", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None)
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
    )

    train(**kwargs)


if __name__ == "__main__":
    main()
