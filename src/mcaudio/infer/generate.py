#!/usr/bin/env python3
"""
generate.py
───────────
Generate audio samples from a base (or LoRA-adapted) AudioLDM2 model.

Usage
─────
    python -m src.mcaudio.infer.generate \\
        --prompt "minecraft zombie hurt sound effect" \\
        --config configs/demo1.yaml

    python -m src.mcaudio.infer.generate \\
        --prompt "minecraft skeleton death sound effect" \\
        --num_samples 4 --output outputs/demo1/baseline
"""
from __future__ import annotations

import argparse
import logging
import os
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


def seed_everything(seed: int) -> torch.Generator:
    """Set global seeds and return a torch Generator for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def generate(
    prompt: str,
    model_id: str = "cvssp/audioldm2",
    lora_weights: str | None = None,
    num_samples: int = 8,
    audio_length_in_s: float = 4.0,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    seed: int = 42,
    output_dir: str = "outputs/demo1/baseline",
) -> list[Path]:
    """
    Generate audio samples and save as .wav files.

    Returns list of saved paths.
    """
    from diffusers import AudioLDM2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    log.info("Loading AudioLDM2 pipeline: %s", model_id)
    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    # ── Work around transformers ≥ 4.45 returning ModelOutput instead of
    #    plain tensors, which breaks AudioLDM2's prompt_embeds[:, None, :].
    #    Force all sub-models to return tuples.  Harmless on older versions.
    for submodel in ("text_encoder", "text_encoder_2", "language_model"):
        m = getattr(pipe, submodel, None)
        if m is not None and hasattr(m, "config"):
            try:
                m.config.return_dict = False
            except Exception:
                pass

    # Optionally load LoRA adapter
    if lora_weights and Path(lora_weights).exists():
        log.info("Loading LoRA weights from %s", lora_weights)
        pipe.unet.load_attn_procs(lora_weights)

    generator = seed_everything(seed)

    log.info("Generating %d samples for prompt: '%s'", num_samples, prompt)

    audios = pipe(
        prompt,
        num_waveforms_per_prompt=num_samples,
        audio_length_in_s=audio_length_in_s,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).audios  # list of numpy arrays

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    # Create a sanitised prefix from the prompt
    prefix = prompt.replace(" ", "_")[:60]
    for i, audio in enumerate(audios):
        fname = out_path / f"{prefix}_{i:03d}.wav"
        sf.write(str(fname), audio, samplerate=16_000, subtype="FLOAT")
        log.info("  saved %s  (%.2f s)", fname.name, len(audio) / 16_000)
        saved.append(fname)

    return saved


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate audio from AudioLDM2.")
    p.add_argument("--prompt", required=True, help="Text prompt for generation.")
    p.add_argument("--config", type=str, default=None, help="YAML config file.")
    p.add_argument("--model_id", type=str, default=None, help="HF model id.")
    p.add_argument("--lora_weights", type=str, default=None, help="Path to LoRA adapter weights.")
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--output", type=str, default=None, help="Output directory.")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Defaults
    kwargs: dict = dict(
        model_id="cvssp/audioldm2",
        num_samples=8,
        audio_length_in_s=4.0,
        num_inference_steps=50,
        guidance_scale=3.5,
        seed=42,
        output_dir="outputs/demo1/baseline",
    )

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        inf = cfg.get("inference", {})
        kwargs.update({
            "model_id": inf.get("model_id", kwargs["model_id"]),
            "num_samples": inf.get("num_samples", kwargs["num_samples"]),
            "audio_length_in_s": inf.get("audio_length_in_s", kwargs["audio_length_in_s"]),
            "num_inference_steps": inf.get("num_inference_steps", kwargs["num_inference_steps"]),
            "guidance_scale": inf.get("guidance_scale", kwargs["guidance_scale"]),
            "seed": inf.get("seed", kwargs["seed"]),
        })
        kwargs["output_dir"] = cfg.get("paths", {}).get("baseline_outputs", kwargs["output_dir"])

    # CLI overrides
    if args.model_id:
        kwargs["model_id"] = args.model_id
    if args.lora_weights:
        kwargs["lora_weights"] = args.lora_weights
    if args.num_samples is not None:
        kwargs["num_samples"] = args.num_samples
    if args.output:
        kwargs["output_dir"] = args.output
    if args.seed is not None:
        kwargs["seed"] = args.seed

    generate(prompt=args.prompt, **kwargs)


if __name__ == "__main__":
    main()
