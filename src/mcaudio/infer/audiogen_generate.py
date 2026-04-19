#!/usr/bin/env python3
"""
audiogen_generate.py
────────────────────
Generate audio samples from a base (or LoRA-adapted) AudioGen model.

Usage
─────
    # Baseline (vanilla AudioGen)
    python -m src.mcaudio.infer.audiogen_generate \\
        --prompt "minecraft zombie hurt sound effect" \\
        --config configs/demo1.yaml

    # With LoRA weights
    python -m src.mcaudio.infer.audiogen_generate \\
        --prompt "minecraft zombie hurt sound effect" \\
        --config configs/demo1.yaml \\
        --lora_weights outputs/audiogen/lora_weights/best
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate(
    prompt: str | list[str],
    model_id: str = "facebook/audiogen-medium",
    lora_weights: str | None = None,
    lora_rank: int = 128,
    lora_alpha: float = 256,
    lora_dropout: float = 0.05,
    lora_targets: list[str] | None = None,
    num_samples: int = 4,
    duration_s: float = 4.0,
    use_sampling: bool = True,
    top_k: int = 250,
    temperature: float = 0.9,
    cfg_coef: float = 3.0,
    seed: int = 42,
    output_dir: str = "outputs/audiogen",
) -> list[Path]:
    """
    Generate audio samples from AudioGen, optionally with LoRA adaptation.

    Returns list of saved .wav file paths.
    """
    if lora_targets is None:
        lora_targets = ["out_proj", "linear1", "linear2"]

    seed_everything(seed)

    log.info("Loading AudioGen: %s", model_id)
    from audiocraft.models import AudioGen

    model = AudioGen.get_pretrained(model_id)
    # Ensure FP32 to avoid dtype mismatch (AudioCraft NaN bug with half)
    model.lm.float()
    model.compression_model.float()
    model.set_generation_params(
        duration=duration_s,
        use_sampling=use_sampling,
        top_k=top_k,
        top_p=0.0,
        temperature=temperature,
        cfg_coef=cfg_coef,
    )

    # ── Optionally load LoRA weights ────────────────────────────────
    if lora_weights:
        lora_path = Path(lora_weights)
        if lora_path.exists():
            log.info("Injecting LoRA adapters and loading weights from %s", lora_path)
            from src.mcaudio.train.audiogen_lora_train import (
                inject_lora,
                load_lora_weights,
            )

            inject_lora(
                model.lm, lora_targets,
                rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
            )
            load_lora_weights(model.lm, lora_path)
            # Move LoRA layers to same device/dtype as the rest of the model
            device = next(model.lm.parameters()).device
            model.lm.to(device).float()
            model.lm.eval()
            log.info("LoRA adapter loaded into AudioGen LM")
        else:
            log.warning("LoRA weights path not found: %s — running baseline", lora_path)

    # ── Generate ────────────────────────────────────────────────────
    if isinstance(prompt, str):
        descriptions = [prompt] * num_samples
    else:
        descriptions = prompt

    log.info("Generating %d samples for: %s", len(descriptions), descriptions[:3])

    with torch.no_grad():
        wav = model.generate(descriptions)  # (B, 1, T) tensor

    # ── Save outputs ────────────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    sr = model.sample_rate

    # Sanitise prompt for filename
    if isinstance(prompt, str):
        prefix = prompt.replace(" ", "_")[:60]
    else:
        prefix = prompt[0].replace(" ", "_")[:60]

    for i in range(wav.shape[0]):
        audio_np = wav[i].cpu().squeeze().numpy()
        fname = out_path / f"{prefix}_{i:03d}.wav"
        sf.write(str(fname), audio_np, samplerate=sr, subtype="FLOAT")
        log.info("  saved %s  (%.2f s)", fname.name, len(audio_np) / sr)
        saved.append(fname)

    return saved


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate audio from AudioGen.")
    p.add_argument("--prompt", required=True, help="Text prompt for generation.")
    p.add_argument("--config", type=str, default=None, help="YAML config file.")
    p.add_argument("--model_id", type=str, default=None)
    p.add_argument("--lora_weights", type=str, default=None, help="Path to LoRA weights.")
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--output", type=str, default=None, help="Output directory.")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Defaults
    kwargs: dict = dict(
        model_id="facebook/audiogen-medium",
        num_samples=4,
        duration_s=4.0,
        use_sampling=True,
        top_k=250,
        temperature=0.9,
        cfg_coef=3.0,
        seed=42,
        output_dir="outputs/audiogen",
        lora_rank=128,
        lora_alpha=256,
        lora_dropout=0.05,
        lora_targets=["out_proj", "linear1", "linear2"],
    )

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        ag = cfg.get("audiogen", {})
        ag_inf = ag.get("inference", {})
        ag_lora = ag.get("lora", {})

        kwargs.update({
            "model_id": ag.get("model_id", kwargs["model_id"]),
            "num_samples": ag_inf.get("num_samples", kwargs["num_samples"]),
            "duration_s": ag_inf.get("duration_s", kwargs["duration_s"]),
            "use_sampling": ag_inf.get("use_sampling", kwargs["use_sampling"]),
            "top_k": ag_inf.get("top_k", kwargs["top_k"]),
            "temperature": ag_inf.get("temperature", kwargs["temperature"]),
            "cfg_coef": ag_inf.get("cfg_coef", kwargs["cfg_coef"]),
            "seed": ag_inf.get("seed", kwargs["seed"]),
            "output_dir": ag.get("paths", {}).get("outputs", kwargs["output_dir"]),
            "lora_rank": ag_lora.get("rank", kwargs["lora_rank"]),
            "lora_alpha": ag_lora.get("alpha", kwargs["lora_alpha"]),
            "lora_dropout": ag_lora.get("dropout", kwargs["lora_dropout"]),
            "lora_targets": ag_lora.get("target_modules", kwargs["lora_targets"]),
        })

    # CLI overrides
    if args.model_id:
        kwargs["model_id"] = args.model_id
    if args.num_samples:
        kwargs["num_samples"] = args.num_samples
    if args.output:
        kwargs["output_dir"] = args.output
    if args.seed:
        kwargs["seed"] = args.seed

    kwargs["prompt"] = args.prompt
    kwargs["lora_weights"] = args.lora_weights

    generate(**kwargs)


if __name__ == "__main__":
    main()
