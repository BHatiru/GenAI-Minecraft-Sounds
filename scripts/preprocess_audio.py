#!/usr/bin/env python3
"""
preprocess_audio.py — Build augmented Minecraft sound dataset
═════════════════════════════════════════════════════════════

Pipeline overview
─────────────────
1. **Load**    – read every raw .ogg → trim silence → normalise → in-memory blocks.
2. **Compose** – combine blocks into meaningful sequences:
     • mob action variant sequences  (hurt1 + hurt2 + …)
     • mob hurt → death combos
     • combat encounters             (mob idle → player damage → mob hurt → death)
     • player step walks / runs      (per surface)
     • mob-specific step walks / runs
     • damage hit sequences + death combos
     • ambient pads
3. **Augment** – speed variations, gap variations for each composed sequence.
4. **Export**  – write 4 s, 16 kHz mono .wav clips + ``_captions.json`` sidecar.

The sidecar file is consumed by ``build_manifest.py`` to produce the final
``metadata.csv`` with train / val splits.

Usage
─────
    python scripts/preprocess_audio.py --config configs/demo1.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Data structure
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SoundBlock:
    """One trimmed & peak-normalised individual sound kept in memory."""
    audio:    np.ndarray   # float32, at target SR
    category: str          # mob | step | damage | ambient
    entity:   str          # zombie | cave | "" …
    action:   str          # hurt | say | death | hit | grass …
    variant:  int          # numeric suffix  (0 when absent)
    source:   str          # relative path for logging

    @property
    def key(self) -> tuple[str, str, str]:
        """Grouping key: (category, entity, action)."""
        return (self.category, self.entity, self.action)


# ═══════════════════════════════════════════════════════════════════
#  Phase 1 · Load raw sounds
# ═══════════════════════════════════════════════════════════════════

_STEM_RE = re.compile(r"^([a-z_]+?)(\d+)?$")


def _parse_ogg_path(rel: Path) -> tuple[str, str, str, int]:
    """Parse relative .ogg path → (category, entity, action, variant)."""
    parts = rel.parts
    stem = rel.stem

    m = _STEM_RE.match(stem)
    action  = m.group(1) if m else stem
    variant = int(m.group(2)) if m and m.group(2) else 0

    if len(parts) >= 3 and parts[0] == "mob":
        return ("mob", parts[1], action, variant)
    if len(parts) >= 3 and parts[0] == "ambient":
        return ("ambient", parts[1], action, variant)
    if len(parts) >= 2:
        return (parts[0], "", action, variant)
    return ("unknown", "", action, variant)


def load_all_blocks(raw_dir: Path, sr: int, top_db: float) -> list[SoundBlock]:
    """Load every .ogg under *raw_dir*, trim silence, peak-normalise."""
    blocks: list[SoundBlock] = []
    for ogg in sorted(raw_dir.rglob("*.ogg")):
        rel = ogg.relative_to(raw_dir)
        cat, ent, act, var = _parse_ogg_path(rel)

        y, _ = librosa.load(ogg, sr=sr, mono=True)
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        if len(yt) < sr * 0.02:          # < 20 ms after trim → keep original
            yt = y
        peak = np.max(np.abs(yt))
        if peak > 0:
            yt = yt / peak
        blocks.append(SoundBlock(yt.astype(np.float32),
                                  cat, ent, act, var, str(rel)))
    log.info("Loaded %d sound blocks from %s", len(blocks), raw_dir)
    return blocks


# ═══════════════════════════════════════════════════════════════════
#  Assembly helpers
# ═══════════════════════════════════════════════════════════════════

def _concat(clips: list[np.ndarray], gap_s: float, sr: int) -> np.ndarray:
    """Concatenate audio clips with silence gaps in between."""
    if not clips:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(sr * gap_s), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, c in enumerate(clips):
        parts.append(c)
        if i < len(clips) - 1:
            parts.append(gap)
    return np.concatenate(parts)


def _fit(seq: np.ndarray, target: int, loop_gap: int = 0) -> np.ndarray:
    """Pad (if ≥ 60 % filled) or loop *seq* to exactly *target* samples."""
    if len(seq) >= target:
        return seq[:target]
    if len(seq) == 0:
        return np.zeros(target, dtype=np.float32)
    if len(seq) >= target * 0.6:
        out = np.zeros(target, dtype=np.float32)
        out[:len(seq)] = seq
        return out
    # Loop with gap
    unit = (np.concatenate([seq, np.zeros(loop_gap, dtype=np.float32)])
            if loop_gap > 0 else seq)
    if len(unit) == 0:
        return np.zeros(target, dtype=np.float32)
    return np.tile(unit, (target // len(unit)) + 1)[:target]


def _speed(y: np.ndarray, factor: float) -> np.ndarray:
    """Change playback speed by *factor* (>1 = faster, <1 = slower).
    Uses linear interpolation — fast and adequate for game SFX."""
    if abs(factor - 1.0) < 0.01:
        return y
    new_len = max(1, int(len(y) / factor))
    old_idx = np.arange(len(y))
    new_idx = np.linspace(0, len(y) - 1, new_len)
    return np.interp(new_idx, old_idx, y).astype(np.float32)


def _norm(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    return (y / peak).astype(np.float32) if peak > 0 else y.astype(np.float32)


def _clip(seq: np.ndarray, sr: int, clip_s: float,
          loop_gap_s: float = 0.3) -> np.ndarray:
    """Pad / trim / loop to exact *clip_s*, then peak-normalise."""
    return _norm(_fit(seq, int(sr * clip_s), int(sr * loop_gap_s)))


# ═══════════════════════════════════════════════════════════════════
#  Augmentation tables
# ═══════════════════════════════════════════════════════════════════
# Each row: (speed_factor, gap_multiplier, file_suffix, caption_modifier)

AUG_FULL = [                       # multi-variant sequences
    (1.0,  1.0, "",       ""),
    (0.85, 1.0, "_slow",  "slow "),
    (1.2,  1.0, "_fast",  "fast "),
    (1.0,  0.5, "_tgap",  "rapid-fire "),
    (1.0,  1.5, "_wgap",  "spaced-out "),
]
AUG_SPEED = [                      # single-variant (no gap to vary)
    (1.0,  1.0, "",       ""),
    (0.85, 1.0, "_slow",  "slow "),
    (1.2,  1.0, "_fast",  "fast "),
]
AUG_STEP = [                       # player footstep surfaces
    (1.0,  1.0, "_walk",      "footsteps walking on "),
    (0.85, 1.0, "_walk_slow", "slow footsteps on "),
    (1.2,  1.0, "_walk_fast", "quick footsteps on "),
    (1.5,  0.3, "_run",       "running footsteps on "),
]
AUG_MOBSTEP = [                    # mob-specific footsteps
    (1.0,  1.0, "_walk",      " walking"),
    (0.85, 1.0, "_walk_slow", " walking slowly"),
    (1.2,  1.0, "_walk_fast", " walking quickly"),
    (1.5,  0.3, "_run",       " running"),
]


# ═══════════════════════════════════════════════════════════════════
#  Caption helpers
# ═══════════════════════════════════════════════════════════════════

_ACT = {
    "say": "growling", "idle": "idling", "breathe": "breathing",
    "moan": "moaning", "scream": "screaming", "stare": "staring",
    "portal": "teleporting", "charge": "charging fireball",
    "fireball": "shooting fireball", "affectionate_scream": "yelping",
    "hurt": "getting hurt", "hit": "getting hit",
    "infect": "infecting", "unfect": "curing", "remedy": "being cured",
    "metal": "banging metal", "wood": "banging wood",
    "woodbreak": "breaking wood",
    "fallbig": "heavy landing", "fallsmall": "light landing",
    "cave": "cave ambience", "enter": "water splash entering",
    "exit": "water splash exiting",
    "underwater_ambience": "underwater ambience",
    "rain": "rain", "thunder": "thunder clap",
}


def _act(action: str) -> str:
    return _ACT.get(action, action.replace("_", " "))


# ═══════════════════════════════════════════════════════════════════
#  Phase 2 + 3 · Compose & Augment
# ═══════════════════════════════════════════════════════════════════

# A produced clip: (relative_path, audio_array, caption)
Clip = tuple[str, np.ndarray, str]


def _augment(
    clips: list[np.ndarray],
    base_gap: float,
    sr: int,
    clip_s: float,
    augments: list[tuple],
    path_prefix: str,
    cap_template: str,           # must contain {mod}
) -> list[Clip]:
    """Apply augmentation table to a list of audio clips.
    Returns one Clip per augment row."""
    out: list[Clip] = []
    for spd, gmul, suffix, mod in augments:
        gap = base_gap * gmul
        seq = _concat(clips, gap, sr)
        if spd != 1.0:
            seq = _speed(seq, spd)
        y = _clip(seq, sr, clip_s, loop_gap_s=gap)
        out.append((f"{path_prefix}{suffix}.wav", y,
                     cap_template.format(mod=mod)))
    return out


# ── Mob sequences ──────────────────────────────────────────────────

def gen_mob(
    blocks: list[SoundBlock],
    damage_blks: list[SoundBlock],
    sr: int,
    clip_s: float,
    mob_gap: float,
    step_gap: float,
    rng: random.Random,
) -> list[Clip]:
    """
    Produce:
      • action-variant sequences with augmentations
      • action → death combos
      • mob step walk / run
      • full combat encounter sequences
    """
    mob = [b for b in blocks if b.category == "mob"]

    # entity → action → [blocks]
    ents: dict[str, dict[str, list[SoundBlock]]] = {}
    for b in mob:
        ents.setdefault(b.entity, {}).setdefault(b.action, []).append(b)
    for e in ents.values():
        for a in e.values():
            a.sort(key=lambda b: b.variant)

    dmg_hits = [b.audio for b in damage_blks if b.action == "hit"]

    results: list[Clip] = []

    for entity, actions in sorted(ents.items()):
        death_audio = actions["death"][0].audio if "death" in actions else None

        for act, blks in sorted(actions.items()):
            if act == "death":
                continue                             # used only in combos

            clips = [b.audio for b in blks]
            prefix = f"mob/{entity}/{act}"
            friendly = _act(act)

            # ── mob-specific footsteps → walk / run variants ──
            if act == "step":
                results.extend(_augment(
                    clips, step_gap, sr, clip_s, AUG_MOBSTEP, prefix,
                    f"minecraft {entity}{{mod}} footsteps sound effect",
                ))
                continue

            # ── action-variant sequence ──
            aug_tbl = AUG_FULL if len(blks) >= 2 else AUG_SPEED
            results.extend(_augment(
                clips, mob_gap, sr, clip_s, aug_tbl,
                prefix + "_seq",
                f"{{mod}}minecraft {entity} {friendly} sound effect",
            ))

            # ── action + death combo (multi-variant actions only) ──
            if death_audio is not None and len(blks) >= 2:
                combo = [rng.choice(clips), death_audio]
                for spd, _, sfx, mod in AUG_SPEED:
                    seq = _concat(combo, mob_gap, sr)
                    if spd != 1.0:
                        seq = _speed(seq, spd)
                    y = _clip(seq, sr, clip_s, loop_gap_s=mob_gap)
                    cap = (f"{mod}minecraft {entity} {friendly} "
                           f"and dying sound effect")
                    results.append((f"{prefix}_death{sfx}.wav", y, cap))

        # ── encounter: idle → player_damage → mob_hurt → death ──
        idle_act = next(
            (a for a in ("say", "idle", "breathe", "moan", "scream")
             if a in actions), None)
        hurt_act = next(
            (a for a in ("hurt", "hit") if a in actions), None)

        if idle_act and hurt_act and death_audio is not None and dmg_hits:
            idle_c = rng.choice([b.audio for b in actions[idle_act]])
            hurt_c = rng.choice([b.audio for b in actions[hurt_act]])
            dmg_c  = rng.choice(dmg_hits)
            parts  = [idle_c, dmg_c, hurt_c, death_audio]
            for spd, _, sfx, mod in AUG_SPEED:
                gap = mob_gap * 0.8
                seq = _concat(parts, gap, sr)
                if spd != 1.0:
                    seq = _speed(seq, spd)
                y = _clip(seq, sr, clip_s, loop_gap_s=gap)
                cap = (f"{mod}minecraft {entity} combat encounter "
                       f"with player damage sound effect")
                results.append(
                    (f"combat/{entity}_encounter{sfx}.wav", y, cap))

        # ── Ghast special: moan → charge → fireball → (delay) → death ──
        if entity == "ghast":
            ghast_a = actions
            if all(a in ghast_a for a in ("moan", "charge", "fireball", "death")):
                moan_c  = rng.choice([b.audio for b in ghast_a["moan"]])
                charge_c = ghast_a["charge"][0].audio
                fb_c     = ghast_a["fireball"][0].audio
                death_c  = ghast_a["death"][0].audio
                # Build with a longer pause before death (player retaliates)
                delay = np.zeros(int(sr * 1.2), dtype=np.float32)
                parts = [moan_c, charge_c, fb_c, delay, death_c]
                for spd, _, sfx, mod in AUG_SPEED:
                    g = mob_gap * 0.6
                    seq = _concat(parts, g, sr)
                    if spd != 1.0:
                        seq = _speed(seq, spd)
                    y = _clip(seq, sr, clip_s, loop_gap_s=g)
                    cap = (f"{mod}minecraft ghast moaning then "
                           f"charging and shooting fireball "
                           f"then dying sound effect")
                    results.append(
                        (f"combat/ghast_fireball{sfx}.wav", y, cap))

    return results


# ── Player step sequences ─────────────────────────────────────────

def gen_steps(
    blocks: list[SoundBlock],
    sr: int,
    clip_s: float,
    step_gap: float,
) -> list[Clip]:
    """Walk and run sequences for every surface type."""
    step_blks = [b for b in blocks if b.category == "step"]
    groups: dict[str, list[SoundBlock]] = {}
    for b in step_blks:
        groups.setdefault(b.action, []).append(b)

    results: list[Clip] = []
    for surface, blks in sorted(groups.items()):
        blks.sort(key=lambda b: b.variant)
        clips = [b.audio for b in blks]
        results.extend(_augment(
            clips, step_gap, sr, clip_s, AUG_STEP,
            f"step/{surface}",
            f"minecraft {{mod}}{surface} surface sound effect",
        ))
    return results


# ── Damage & cross-category combos ────────────────────────────────

def gen_damage(
    blocks: list[SoundBlock],
    mob_blocks: list[SoundBlock],
    sr: int,
    clip_s: float,
    gap: float,
    rng: random.Random,
) -> list[Clip]:
    """
    Produce:
      • damage hit / fall sequences with augmentations
      • hit + mob_death cross-category combos
    """
    dmg = [b for b in blocks if b.category == "damage"]
    groups: dict[str, list[SoundBlock]] = {}
    for b in dmg:
        groups.setdefault(b.action, []).append(b)

    # Collect mob death sounds for combos
    mob_deaths = {b.entity: b.audio
                  for b in mob_blocks if b.action == "death"}

    results: list[Clip] = []

    for act, blks in sorted(groups.items()):
        blks.sort(key=lambda b: b.variant)
        clips = [b.audio for b in blks]
        friendly = _act(act)

        aug_tbl = AUG_FULL if len(blks) >= 2 else AUG_SPEED
        results.extend(_augment(
            clips, gap, sr, clip_s, aug_tbl,
            f"damage/{act}_seq",
            f"{{mod}}minecraft player {friendly} sound effect",
        ))

        # hit + mob_death combos (3 representative mobs)
        if act == "hit":
            for mob_name in ("zombie", "skeleton", "creeper"):
                if mob_name not in mob_deaths:
                    continue
                combo = [rng.choice(clips), rng.choice(clips),
                         mob_deaths[mob_name]]
                for spd, _, sfx, mod in AUG_SPEED:
                    seq = _concat(combo, gap, sr)
                    if spd != 1.0:
                        seq = _speed(seq, spd)
                    y = _clip(seq, sr, clip_s, loop_gap_s=gap)
                    cap = (f"{mod}minecraft player taking damage "
                           f"and {mob_name} dying sound effect")
                    results.append(
                        (f"combat/damage_{mob_name}_death{sfx}.wav", y, cap))

    return results


# ── Ambient pads ───────────────────────────────────────────────────

def gen_ambient(
    blocks: list[SoundBlock],
    sr: int,
    clip_s: float,
) -> list[Clip]:
    """Individual ambient clips padded to length, plus slow variant."""
    amb = [b for b in blocks if b.category == "ambient"]
    target = int(sr * clip_s)
    results: list[Clip] = []

    for b in amb:
        y = _norm(_fit(b.audio, target))
        vs = str(b.variant) if b.variant else ""
        fname = f"{b.action}{vs}"
        base_path = (f"ambient/{b.entity}/{fname}"
                     if b.entity else f"ambient/{fname}")
        friendly = _act(b.action)

        results.append((f"{base_path}.wav", y,
                         f"minecraft {friendly} sound effect"))

        # Slow variant
        ys = _speed(y, 0.85)
        ys = _norm(_fit(ys, target))
        results.append((f"{base_path}_slow.wav", ys,
                         f"slow minecraft {friendly} sound effect"))

    return results


# ═══════════════════════════════════════════════════════════════════
#  Phase 4 · Export
# ═══════════════════════════════════════════════════════════════════

def export_all(seqs: list[Clip], out_dir: Path, sr: int) -> int:
    """Write every clip as .wav and produce ``_captions.json`` sidecar."""
    captions: dict[str, str] = {}
    ok = 0
    for rel, audio, cap in seqs:
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), audio, sr, subtype="FLOAT")
        captions[rel] = cap
        ok += 1

    cap_path = out_dir / "_captions.json"
    with open(cap_path, "w", encoding="utf-8") as fh:
        json.dump(captions, fh, indent=2, ensure_ascii=False)
    log.info("Wrote caption sidecar → %s  (%d entries)", cap_path, len(captions))
    return ok


# ═══════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════

def process_all(
    raw_dir: Path,
    out_dir: Path,
    sr: int = 16_000,
    clip_s: float = 4.0,
    top_db: float = 30.0,
    mob_gap: float = 0.7,
    step_gap: float = 0.1,
    damage_gap: float = 0.5,
    seed: int = 42,
) -> int:
    """Full pipeline: load → compose → augment → export."""
    rng = random.Random(seed)

    # Phase 1
    blocks = load_all_blocks(raw_dir, sr, top_db)
    mob_blks = [b for b in blocks if b.category == "mob"]
    dmg_blks = [b for b in blocks if b.category == "damage"]

    # Phases 2 + 3
    all_clips: list[Clip] = []

    log.info("── Generating mob sequences …")
    c = gen_mob(blocks, dmg_blks, sr, clip_s,
                mob_gap=mob_gap, step_gap=step_gap, rng=rng)
    all_clips.extend(c)
    log.info("   → %d mob clips", len(c))

    log.info("── Generating step sequences …")
    c = gen_steps(blocks, sr, clip_s, step_gap=step_gap)
    all_clips.extend(c)
    log.info("   → %d step clips", len(c))

    log.info("── Generating damage / combat sequences …")
    c = gen_damage(blocks, mob_blks, sr, clip_s,
                   gap=damage_gap, rng=rng)
    all_clips.extend(c)
    log.info("   → %d damage/combat clips", len(c))

    log.info("── Generating ambient clips …")
    c = gen_ambient(blocks, sr, clip_s)
    all_clips.extend(c)
    log.info("   → %d ambient clips", len(c))

    log.info("── Total: %d clips to export", len(all_clips))

    # Phase 4
    if out_dir.exists():
        shutil.rmtree(out_dir)
        log.info("Cleared previous output directory")

    ok = export_all(all_clips, out_dir, sr)

    durs = [len(a) / sr for _, a, _ in all_clips]
    log.info(
        "Done – %d clips exported  |  %.2f–%.2f s  |  mean %.2f s",
        ok, min(durs), max(durs), np.mean(durs),
    )
    return ok


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Build augmented Minecraft sound dataset")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--input",  type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--sr",     type=int, default=None)
    ap.add_argument("--dur",    type=float, default=None)
    args = ap.parse_args(argv)

    # Defaults
    raw_dir = "data/raw"
    out_dir = "data/processed"
    sr = 16_000
    clip_s = 4.0
    top_db = 30.0
    mob_gap = 0.7
    step_gap = 0.1
    damage_gap = 0.5
    seed = 42

    if args.config:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
        raw_dir    = cfg.get("paths", {}).get("raw_assets", raw_dir)
        out_dir    = cfg.get("paths", {}).get("processed", out_dir)
        ac         = cfg.get("audio", {})
        sr         = ac.get("sample_rate", sr)
        clip_s     = ac.get("clip_length_s", clip_s)
        top_db     = ac.get("silence_top_db", top_db)
        mob_gap    = ac.get("mob_gap_s", mob_gap)
        step_gap   = ac.get("step_gap_s", step_gap)
        damage_gap = ac.get("damage_gap_s", damage_gap)
        seed       = cfg.get("split", {}).get("seed", seed)

    if args.input:  raw_dir = args.input
    if args.output: out_dir = args.output
    if args.sr:     sr = args.sr
    if args.dur:    clip_s = args.dur

    raw_p = Path(raw_dir)
    if not raw_p.exists():
        log.error("Input directory missing: %s", raw_p)
        sys.exit(1)

    process_all(
        raw_p, Path(out_dir),
        sr=sr, clip_s=clip_s, top_db=top_db,
        mob_gap=mob_gap, step_gap=step_gap,
        damage_gap=damage_gap, seed=seed,
    )


if __name__ == "__main__":
    main()
