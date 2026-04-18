#!/usr/bin/env python3
"""
prepare_audiogen_data.py
────────────────────────
Convert the existing Minecraft audio dataset into AudioCraft's expected format.

Reads ``data/manifest.csv`` and produces:
  1. JSON sidecar files next to each .wav  (``<name>.wav.json``)
     with ``{"description": "<caption>"}``
  2. JSONL manifest files for train / valid splits
     (``data/audiogen/train.jsonl``, ``data/audiogen/valid.jsonl``)

Usage
─────
    python scripts/prepare_audiogen_data.py --config configs/demo1.yaml
    python scripts/prepare_audiogen_data.py \
        --manifest data/manifest.csv \
        --processed_dir data/processed \
        --output_dir data/audiogen
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import soundfile as sf
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


def prepare(
    manifest_csv: str,
    processed_dir: str,
    output_dir: str,
) -> None:
    proc = Path(processed_dir).resolve()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict]] = {"train": [], "val": []}

    with open(manifest_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["file_name"]
            caption = row["caption"]
            split = row["split"]        # "train" or "val"

            wav_path = proc / fname
            if not wav_path.exists():
                log.warning("Missing file, skipping: %s", wav_path)
                continue

            # ── 1. Write JSON sidecar next to the .wav ──────────────
            sidecar = wav_path.parent / f"{wav_path.name}.json"
            sidecar.write_text(
                json.dumps({"description": caption}, ensure_ascii=False),
                encoding="utf-8",
            )

            # ── 2. Collect JSONL entry ──────────────────────────────
            info = sf.info(str(wav_path))
            entry = {
                "path": str(wav_path),
                "duration": round(info.duration, 4),
                "sample_rate": info.samplerate,
            }
            splits.setdefault(split, []).append(entry)

    # ── 3. Write JSONL manifests ────────────────────────────────────
    for split_name, entries in splits.items():
        jsonl_path = out / f"{split_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fp:
            for entry in entries:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info("Wrote %s  (%d entries)", jsonl_path, len(entries))

    total = sum(len(v) for v in splits.values())
    log.info("Done — %d total entries, JSON sidecars written next to .wav files", total)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare AudioCraft-format dataset from existing manifest.",
    )
    p.add_argument("--config", type=str, default=None, help="YAML config file.")
    p.add_argument("--manifest", type=str, default=None, help="Path to manifest.csv.")
    p.add_argument("--processed_dir", type=str, default=None, help="Processed .wav dir.")
    p.add_argument("--output_dir", type=str, default=None, help="Output dir for JSONL.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    manifest = args.manifest or "data/manifest.csv"
    processed_dir = args.processed_dir or "data/processed"
    output_dir = args.output_dir or "data/audiogen"

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        paths = cfg.get("paths", {})
        manifest = args.manifest or paths.get("manifest", manifest)
        processed_dir = args.processed_dir or paths.get("processed", processed_dir)
        ag = cfg.get("audiogen", {}).get("paths", {})
        output_dir = args.output_dir or ag.get("dataset", output_dir)

    prepare(manifest, processed_dir, output_dir)


if __name__ == "__main__":
    main()
