#!/usr/bin/env python3
"""
build_manifest.py
─────────────────
Create ``metadata.csv`` from the processed .wav files.

Each row contains:
    file_name   – path to the .wav file (relative to processed dir)
    caption     – simple text caption derived from the folder structure

Caption derivation (from flat file structure):
    Given path  ``mob/zombie/hurt1.wav``
    →  entity = "zombie",  action = "hurt"  (strip trailing digits from filename)
    →  caption = "minecraft zombie hurt sound effect"

    Given path  ``step/grass3.wav``
    →  category = "step",  surface = "grass"
    →  caption = "minecraft grass footstep sound effect"

    Given path  ``damage/hit2.wav``
    →  category = "damage",  action = "hit"
    →  caption = "minecraft player hit damage sound effect"

Also produces a train / val split column.

Usage
─────
    python scripts/build_manifest.py --config configs/demo1.yaml
    python scripts/build_manifest.py --processed data/processed --out data/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Caption helpers ─────────────────────────────────────────────────

import re

# Mapping of filename action prefix → friendlier caption words
ACTION_ALIASES: dict[str, str] = {
    "hurt": "hurt",
    "death": "death",
    "say": "idle groan",
    "step": "footstep",
    "idle": "idle",
    "attack": "attack",
    "shoot": "shooting",
    "infect": "infection",
    "unfect": "cure",
    "remedy": "remedy",
    "metal": "metal hit",
    "wood": "wood hit",
    "woodbreak": "wood breaking",
    # damage category
    "hit": "hit damage",
    "fallbig": "heavy fall damage",
    "fallsmall": "light fall damage",
    # step surfaces
    "cloth": "cloth footstep",
    "coral": "coral footstep",
    "grass": "grass footstep",
    "gravel": "gravel footstep",
    "ladder": "ladder climbing",
    "sand": "sand footstep",
    "scaffold": "scaffolding footstep",
    "snow": "snow footstep",
    "stone": "stone footstep",
    "wet_grass": "wet grass footstep",
    # ambient sounds
    "cave": "cave ambience",
    "enter": "underwater enter",
    "exit": "underwater exit",
    "underwater_ambience": "underwater ambience",
    "rain": "rain",
    "thunder": "thunder",
    # mob-specific
    "breathe": "breathing",
    "scream": "scream",
    "stare": "stare",
    "portal": "teleport",
    "charge": "fireball charge",
    "fireball": "fireball",
    "affectionate_scream": "moan",
    "moan": "moan",
}

# Regex to strip trailing digits from a filename stem: "hurt1" → "hurt"
_STRIP_DIGITS = re.compile(r"(\d+)$")


def _stem_action(filename_stem: str) -> str:
    """Extract the action keyword from a filename stem by stripping trailing digits."""
    return _STRIP_DIGITS.sub("", filename_stem)


def caption_from_path(rel_path: Path, template: str) -> str:
    """
    Derive a caption string from the relative wav path.

    Handles two structures:
      mob/<entity>/<action><N>.wav   →  "minecraft <entity> <action> sound effect"
      <category>/<action><N>.wav     →  "minecraft <action> sound effect"  (damage, step, etc.)

    Parameters
    ----------
    rel_path : Path
        e.g. Path("mob/zombie/hurt1.wav") or Path("step/grass3.wav")
    template : str
        Python format string with ``{entity}`` and ``{action}`` placeholders.

    Returns
    -------
    str  – e.g. "minecraft zombie hurt sound effect"
    """
    parts = rel_path.parts  # e.g. ('mob', 'zombie', 'hurt1.wav')
    stem = rel_path.stem     # e.g. 'hurt1'
    action_raw = _stem_action(stem)

    if len(parts) >= 3 and parts[0] == "mob":
        # mob/<entity_name>/<action><N>.wav
        entity = parts[1]  # 'zombie'
    elif len(parts) >= 3 and parts[0] == "ambient":
        # ambient/<subtype>/<file>.wav  →  entity = subtype (cave, weather, underwater)
        entity = parts[1]  # 'cave', 'weather', 'underwater'
    elif len(parts) >= 2:
        # top-level category like damage/ or step/
        entity = parts[0]  # 'damage' or 'step'
    else:
        entity = "unknown"

    action = ACTION_ALIASES.get(action_raw, action_raw)

    # For step/ and damage/ categories, skip entity in caption
    if entity in ("step", "damage"):
        return f"minecraft {action} sound effect"

    # For ambient sounds, use a descriptive caption
    if parts[0] == "ambient":
        return f"minecraft {action} sound effect"

    return template.format(entity=entity, action=action)


# ── Core ────────────────────────────────────────────────────────────


def _load_caption_sidecar(processed_dir: Path) -> dict[str, str] | None:
    """Try to load ``_captions.json`` produced by preprocess_audio.py."""
    cap_path = processed_dir / "_captions.json"
    if not cap_path.exists():
        return None
    import json
    with open(cap_path, "r", encoding="utf-8") as fh:
        data: dict[str, str] = json.load(fh)
    log.info("Loaded %d captions from %s", len(data), cap_path)
    return data


def build_manifest(
    processed_dir: Path,
    out_csv: Path,
    template: str = "minecraft {entity} {action} sound effect",
    val_fraction: float = 0.15,
    seed: int = 42,
) -> int:
    """
    Walk *processed_dir* for .wav files, generate captions, split, write CSV.

    If a ``_captions.json`` sidecar exists (produced by the augmentation
    pipeline), those captions are used.  Otherwise falls back to
    path-based caption derivation.

    Returns the total number of rows written.
    """
    wav_files = sorted(processed_dir.rglob("*.wav"))
    if not wav_files:
        log.warning("No .wav files in %s", processed_dir)
        return 0

    caption_sidecar = _load_caption_sidecar(processed_dir)

    rows: list[dict[str, str]] = []
    for wav in wav_files:
        rel = wav.relative_to(processed_dir)
        # Normalise to forward-slash key for sidecar lookup
        rel_key = str(rel).replace("\\", "/")

        if caption_sidecar and rel_key in caption_sidecar:
            caption = caption_sidecar[rel_key]
        else:
            caption = caption_from_path(rel, template)
        rows.append({"file_name": rel_key, "caption": caption})

    # Shuffle & split
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    for i, row in enumerate(rows):
        row["split"] = "val" if i < n_val else "train"

    # Sort back by file_name for deterministic output
    rows.sort(key=lambda r: r["file_name"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "caption", "split"])
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val_actual = sum(1 for r in rows if r["split"] == "val")
    log.info("Manifest written to %s  (%d rows: %d train, %d val)", out_csv, len(rows), n_train, n_val_actual)

    # Print a few example rows
    log.info("── Example rows ──")
    for r in rows[:5]:
        log.info("  %s | %s | %s", r["split"], r["file_name"], r["caption"])

    return len(rows)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build metadata.csv manifest for processed audio.")
    p.add_argument("--config", type=str, default=None, help="YAML config file.")
    p.add_argument("--processed", type=str, default=None, help="Processed wav directory.")
    p.add_argument("--out", type=str, default=None, help="Output CSV path.")
    p.add_argument("--val-fraction", type=float, default=None, help="Fraction for validation split.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for split.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    processed_dir = "data/processed"
    out_csv = "data/manifest.csv"
    template = "minecraft {entity} {action} sound effect"
    val_fraction = 0.15
    seed = 42

    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        processed_dir = cfg.get("paths", {}).get("processed", processed_dir)
        out_csv = cfg.get("paths", {}).get("manifest", out_csv)
        template = cfg.get("caption_template", template)
        split_cfg = cfg.get("split", {})
        val_fraction = split_cfg.get("val_fraction", val_fraction)
        seed = split_cfg.get("seed", seed)

    if args.processed:
        processed_dir = args.processed
    if args.out:
        out_csv = args.out
    if args.val_fraction is not None:
        val_fraction = args.val_fraction
    if args.seed is not None:
        seed = args.seed

    processed_path = Path(processed_dir)
    if not processed_path.exists():
        log.error("Processed directory does not exist: %s", processed_path)
        sys.exit(1)

    build_manifest(processed_path, Path(out_csv), template=template, val_fraction=val_fraction, seed=seed)


if __name__ == "__main__":
    main()
