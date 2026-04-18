#!/usr/bin/env python3
"""
audiogen_lora_train.py
──────────────────────
LoRA fine-tuning of AudioGen's language model on Minecraft sound effects.

Pipeline
────────
1. Load facebook/audiogen-medium (EnCodec tokenizer + Transformer LM).
2. Freeze EnCodec and T5 text conditioner entirely.
3. Inject LoRA adapters into the LM transformer's out_proj, linear1,
   linear2 layers (NOT q/k/v — they are fused into in_proj_weight).
4. For each training step:
     a. Encode waveforms → EnCodec discrete codes  [B, K=4, T].
     b. Build ConditioningAttributes from text descriptions.
     c. lm.compute_predictions(codes, conditions) → logits + mask.
     d. nan_to_num on logits (delay-pattern produces NaN at offsets).
     e. Masked cross-entropy loss.
5. Save LoRA adapter weights.

Uses FP32 throughout to avoid AudioCraft's known NaN bug with autocast.
Designed for a single GPU (T4 16 GB / RTX 5090 32 GB).

Usage
─────
    python -m src.mcaudio.train.audiogen_lora_train --config configs/demo1.yaml
    python -m src.mcaudio.train.audiogen_lora_train --config configs/demo1.yaml --epochs 50
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import logging
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
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
    Waveforms are preprocessed to 16 kHz mono float32.
    """

    def __init__(
        self,
        manifest_csv: str,
        processed_dir: str,
        split: str = "train",
        sample_rate: int = 16_000,
    ):
        self.processed_dir = Path(processed_dir)
        self.sample_rate = sample_rate
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
        # Ensure correct shape: (channels, samples) — mono → (1, T)
        wav = torch.from_numpy(audio).unsqueeze(0)
        return {"audio": wav, "caption": row["caption"]}


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length waveforms by padding to max length."""
    max_len = max(item["audio"].shape[-1] for item in batch)
    wavs = []
    for item in batch:
        wav = item["audio"]
        pad = max_len - wav.shape[-1]
        if pad > 0:
            wav = F.pad(wav, (0, pad))
        wavs.append(wav)
    return {
        "audio": torch.stack(wavs, dim=0),   # (B, 1, T)
        "captions": [item["caption"] for item in batch],
    }


# ═══════════════════════════════════════════════════════════════════
#  LoRA injection
# ═══════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adaptation."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 128,
        alpha: float = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base = base_linear
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming init for A, zero init for B → LoRA starts as identity
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.dropout(self.lora_A(x)))
        return base_out + self.scale * lora_out


def inject_lora(
    model: nn.Module,
    target_names: list[str],
    rank: int = 128,
    alpha: float = 256,
    dropout: float = 0.05,
) -> int:
    """
    Replace matching nn.Linear layers with LoRA wrappers.

    Returns number of injected adapters.
    """
    count = 0
    for name, module in list(model.named_modules()):
        # Check if the leaf name matches one of the targets
        leaf = name.split(".")[-1]
        if leaf not in target_names:
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Navigate to parent and replace
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], lora_layer)
        count += 1

    return count


def freeze_base_params(model: nn.Module) -> tuple[int, int]:
    """Freeze all non-LoRA parameters. Returns (frozen, trainable) counts."""
    frozen = trainable = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    return frozen, trainable


def save_lora_weights(model: nn.Module, path: str | Path) -> None:
    """Save only the LoRA adapter weights (lora_A, lora_B)."""
    state = {
        name: param.cpu().clone()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, path / "lora_weights.pt")
    log.info("Saved LoRA weights (%d tensors) → %s", len(state), path / "lora_weights.pt")


def load_lora_weights(model: nn.Module, path: str | Path) -> None:
    """Load LoRA adapter weights into an already-injected model."""
    ckpt = Path(path)
    if ckpt.is_dir():
        ckpt = ckpt / "lora_weights.pt"
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    missing, unexpected = [], []
    model_state = dict(model.named_parameters())
    for k, v in state.items():
        if k in model_state:
            model_state[k].data.copy_(v)
        else:
            unexpected.append(k)
    if unexpected:
        log.warning("Unexpected keys in checkpoint: %s", unexpected)
    log.info("Loaded LoRA weights from %s", ckpt)


# ═══════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════

def train(
    config: dict,
    manifest_csv: str,
    processed_dir: str,
    output_dir: str,
    epochs: int = 150,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 3e-4,
    warmup_steps: int = 50,
    weight_decay: float = 1e-5,
    cfg_dropout: float = 0.1,
    save_every: int = 25,
    seed: int = 42,
    lora_rank: int = 128,
    lora_alpha: float = 256,
    lora_dropout: float = 0.05,
    lora_targets: list[str] | None = None,
    model_id: str = "facebook/audiogen-medium",
) -> None:
    if lora_targets is None:
        lora_targets = ["out_proj", "linear1", "linear2"]

    # ── Seed ────────────────────────────────────────────────────────
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load AudioGen ───────────────────────────────────────────────
    log.info("Loading AudioGen: %s", model_id)
    from audiocraft.models import AudioGen

    model = AudioGen.get_pretrained(model_id)
    model.set_generation_params(duration=4.0)

    lm = model.lm           # LanguageModel — the transformer we fine-tune
    encodec = model.compression_model   # EnCodec — frozen tokenizer

    lm.to(device)
    encodec.to(device)

    # ── Freeze everything first ─────────────────────────────────────
    for param in lm.parameters():
        param.requires_grad = False
    for param in encodec.parameters():
        param.requires_grad = False
    encodec.eval()

    # ── Inject LoRA ─────────────────────────────────────────────────
    n_adapters = inject_lora(
        lm, lora_targets,
        rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
    )
    frozen_count, trainable_count = freeze_base_params(lm)
    log.info(
        "LoRA injected: %d adapters | Trainable: %.2f M | Frozen: %.2f M",
        n_adapters,
        trainable_count / 1e6,
        frozen_count / 1e6,
    )

    # ── Dataloaders ─────────────────────────────────────────────────
    train_ds = McAudioDataset(manifest_csv, processed_dir, split="train")
    val_ds = McAudioDataset(manifest_csv, processed_dir, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # ── Optimizer & scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in lm.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=1e-6,
    )

    log.info(
        "Training: %d epochs, batch=%d, grad_accum=%d, effective_batch=%d, "
        "total_steps=%d, lr=%.1e",
        epochs, batch_size, gradient_accumulation_steps,
        batch_size * gradient_accumulation_steps,
        total_steps, learning_rate,
    )

    # ── Import conditioning helpers ─────────────────────────────────
    from audiocraft.modules.conditioners import ConditioningAttributes

    # ── Training loop ───────────────────────────────────────────────
    best_val_loss = float("inf")
    global_step = 0
    warmup_done = False

    for epoch in range(1, epochs + 1):
        lm.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            wav = batch["audio"].to(device)           # (B, 1, T)
            captions = batch["captions"]

            # ── Encode audio → discrete codes ───────────────────
            with torch.no_grad():
                encoded = encodec.encode(wav)
                # encoded is a list of tuples: [(codes, scale), ...]
                # codes shape: (B, K, T_codes)
                codes = encoded[0][0]

            # ── Build conditioning ──────────────────────────────
            conditions = []
            for cap in captions:
                # CFG dropout: replace caption with empty string randomly
                if cfg_dropout > 0 and torch.rand(1).item() < cfg_dropout:
                    cap = ""
                attr = ConditioningAttributes(text={"description": cap})
                conditions.append(attr)

            # ── Forward through LM ──────────────────────────────
            lm_output = lm.compute_predictions(codes, conditions)
            logits = lm_output.logits   # (B, K, T, card)
            mask = lm_output.mask       # (B, K, T)

            # Critical: delay pattern produces NaN for offset positions
            logits = logits.nan_to_num(nan=0.0)

            # ── Masked cross-entropy ────────────────────────────
            B, K, T, card = logits.shape
            # Target codes — trim to match logits time dim
            target = codes[:, :, :T].long()

            logits_flat = logits.reshape(-1, card)
            target_flat = target.reshape(-1)
            mask_flat = mask.reshape(-1)

            loss = F.cross_entropy(
                logits_flat[mask_flat],
                target_flat[mask_flat],
                reduction="mean",
            )

            loss = loss / gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * gradient_accumulation_steps
            epoch_tokens += mask_flat.sum().item()

            # ── Gradient accumulation step ──────────────────────
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in lm.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Linear warmup
                if global_step <= warmup_steps:
                    warmup_lr = learning_rate * global_step / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr
                elif not warmup_done:
                    warmup_done = True
                    for pg in optimizer.param_groups:
                        pg["lr"] = learning_rate

                if warmup_done:
                    scheduler.step()

        # ── Epoch logging ───────────────────────────────────────────
        n_batches = len(train_loader)
        avg_train = epoch_loss / max(n_batches, 1)

        # ── Validation ──────────────────────────────────────────────
        val_loss = _validate(lm, encodec, val_loader, device)

        lr_now = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %3d/%d  |  train_loss=%.4f  val_loss=%.4f  lr=%.2e  step=%d",
            epoch, epochs, avg_train, val_loss, lr_now, global_step,
        )

        # ── Checkpointing ──────────────────────────────────────────
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_dir = Path(output_dir) / f"epoch_{epoch:04d}"
            save_lora_weights(lm, ckpt_dir)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_lora_weights(lm, Path(output_dir) / "best")
            log.info("  ↑ New best val_loss=%.4f — saved to best/", val_loss)

    # ── Final save ──────────────────────────────────────────────────
    save_lora_weights(lm, Path(output_dir) / "final")
    log.info("Training complete. Best val_loss=%.4f", best_val_loss)


@torch.no_grad()
def _validate(lm, encodec, val_loader, device) -> float:
    """Run one validation pass, return average loss."""
    from audiocraft.modules.conditioners import ConditioningAttributes

    lm.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        wav = batch["audio"].to(device)
        captions = batch["captions"]

        encoded = encodec.encode(wav)
        codes = encoded[0][0]

        conditions = [
            ConditioningAttributes(text={"description": cap})
            for cap in captions
        ]

        lm_output = lm.compute_predictions(codes, conditions)
        logits = lm_output.logits.nan_to_num(nan=0.0)
        mask = lm_output.mask

        B, K, T, card = logits.shape
        target = codes[:, :, :T].long()

        logits_flat = logits.reshape(-1, card)
        target_flat = target.reshape(-1)
        mask_flat = mask.reshape(-1)

        loss = F.cross_entropy(
            logits_flat[mask_flat],
            target_flat[mask_flat],
            reduction="mean",
        )
        total_loss += loss.item()
        total_batches += 1

    lm.train()
    return total_loss / max(total_batches, 1)


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune AudioGen LM.")
    p.add_argument("--config", type=str, default="configs/demo1.yaml")
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--processed_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lora_rank", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    ag = cfg.get("audiogen", {})
    ag_train = ag.get("training", {})
    ag_lora = ag.get("lora", {})

    train(
        config=cfg,
        manifest_csv=args.manifest or paths.get("manifest", "data/manifest.csv"),
        processed_dir=args.processed_dir or paths.get("processed", "data/processed"),
        output_dir=args.output_dir or ag.get("paths", {}).get("weights", "outputs/audiogen/lora_weights"),
        epochs=args.epochs or ag_train.get("epochs", 150),
        batch_size=args.batch_size or ag_train.get("batch_size", 2),
        gradient_accumulation_steps=ag_train.get("gradient_accumulation_steps", 4),
        learning_rate=args.lr or ag_train.get("learning_rate", 3e-4),
        warmup_steps=ag_train.get("warmup_steps", 50),
        weight_decay=ag_train.get("weight_decay", 1e-5),
        cfg_dropout=ag_train.get("cfg_dropout", 0.1),
        save_every=ag_train.get("save_every", 25),
        seed=args.seed or ag_train.get("seed", 42),
        lora_rank=args.lora_rank or ag_lora.get("rank", 128),
        lora_alpha=ag_lora.get("alpha", 256),
        lora_dropout=ag_lora.get("dropout", 0.05),
        lora_targets=ag_lora.get("target_modules", ["out_proj", "linear1", "linear2"]),
        model_id=ag.get("model_id", "facebook/audiogen-medium"),
    )


if __name__ == "__main__":
    main()
