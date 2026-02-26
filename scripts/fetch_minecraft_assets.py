#!/usr/bin/env python3
"""
fetch_minecraft_assets.py
─────────────────────────
Download selected Minecraft sound assets (.ogg) from
https://github.com/InventivetalentDev/minecraft-assets (branch 1.21.1).

Uses the GitHub Trees API to list files in each requested subfolder, then
downloads them individually.  No git clone / sparse checkout needed.

Usage
─────
    python scripts/fetch_minecraft_assets.py --config configs/demo1.yaml
    python scripts/fetch_minecraft_assets.py --categories entity/zombie/hurt entity/zombie/ambient --out data/raw
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

REPO_OWNER = "InventivetalentDev"
REPO_NAME = "minecraft-assets"
BRANCH = "1.21.1"
SOUND_PREFIX = "assets/minecraft/sounds"

# ── GitHub helpers ──────────────────────────────────────────────────


def _github_api_tree(path: str) -> list[dict]:
    """Return a flat list of blobs under *path* using the Git Trees API."""
    # First get the SHA of the tree at the path via the Contents API
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}?ref={BRANCH}"
    log.debug("GET %s", url)
    resp = requests.get(url, timeout=30)

    if resp.status_code == 404:
        log.warning("Path not found on GitHub: %s", path)
        return []

    resp.raise_for_status()
    data = resp.json()
    # The Contents API for a directory returns a list of entries.
    if isinstance(data, list):
        return [item for item in data if item.get("type") == "file" and item["name"].endswith(".ogg")]
    return []


def _download_file(url: str, dest: Path) -> None:
    """Download a single file from a raw URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


# ── Core logic ──────────────────────────────────────────────────────


def fetch_category(category: str, out_dir: Path) -> int:
    """
    Download all .ogg files for a single category.

    Parameters
    ----------
    category : str
        e.g. "entity/zombie/hurt"
    out_dir : Path
        Root output directory (files go into out_dir/entity/zombie/hurt/*.ogg).

    Returns
    -------
    int  – number of files downloaded.
    """
    api_path = f"{SOUND_PREFIX}/{category}"
    files = _github_api_tree(api_path)
    if not files:
        log.warning("No .ogg files found for category '%s'", category)
        return 0

    count = 0
    for item in files:
        name = item["name"]
        download_url = item.get("download_url")
        if not download_url:
            continue
        dest = out_dir / category / name
        if dest.exists():
            log.debug("Skipping (exists): %s", dest)
            count += 1
            continue
        log.info("  ↓ %s", f"{category}/{name}")
        _download_file(download_url, dest)
        count += 1
    return count


def fetch_all(categories: List[str], out_dir: Path) -> None:
    total = 0
    for cat in categories:
        log.info("Fetching category: %s", cat)
        n = fetch_category(cat, out_dir)
        log.info("  → %d files", n)
        total += n
    log.info("Total files downloaded: %d", total)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Minecraft sound assets (.ogg) from GitHub.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (reads paths.raw_assets and categories).",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help='Sound categories to download, e.g.  entity/zombie/hurt  (overrides config).',
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for raw assets (overrides config).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    categories: list[str] | None = args.categories
    out_dir: str | None = args.out

    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        if categories is None:
            categories = cfg.get("categories", [])
        if out_dir is None:
            out_dir = cfg.get("paths", {}).get("raw_assets", "data/raw")

    # Fallbacks
    if not categories:
        log.error("No categories specified. Use --categories or --config.")
        sys.exit(1)
    if out_dir is None:
        out_dir = "data/raw"

    fetch_all(categories, Path(out_dir))


if __name__ == "__main__":
    main()
