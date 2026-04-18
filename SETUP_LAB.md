# Lab PC Setup Guide — AudioGen LoRA Fine-Tuning

## Context for AI Agent

This is a GenAI course project (CSCI 595) that fine-tunes audio generation models to produce
Minecraft-style sound effects from text prompts. The **current approach** uses Meta's AudioGen
(autoregressive transformer on EnCodec discrete tokens) with LoRA adapters on the language model.

### What was already done (on another machine)
- Full data pipeline: 195 raw Minecraft .ogg files → preprocessed to 325 augmented 16kHz mono
  4-second .wav clips (277 train / 48 val) with text captions
- All code is written and committed. The new AudioGen approach files are:
  - `scripts/prepare_audiogen_data.py` — converts manifest.csv → AudioCraft format (JSONL + JSON sidecars)
  - `src/mcaudio/train/audiogen_lora_train.py` — LoRA training on AudioGen's LM transformer
  - `src/mcaudio/infer/audiogen_generate.py` — inference with vanilla or LoRA-adapted AudioGen
  - `notebooks/audiogen_finetune.ipynb` — end-to-end notebook (adapted for local GPU)
  - `configs/demo1.yaml` — contains `audiogen:` config block with all hyperparams

### What needs to happen on the lab PC
1. Set up Python environment with `uv`
2. Copy `data/` folder from another machine (or regenerate)
3. Run the notebook `notebooks/audiogen_finetune.ipynb` cell by cell
4. Debug any runtime issues (audiocraft version compat, CUDA, etc.)

---

## Step 1: Clone & Set Up Environment

```powershell
# Clone the repo
git clone https://github.com/BHatiru/GenAI-Minecraft-Sounds.git
cd GenAI-Minecraft-Sounds

# Create a virtual environment with uv (Python 3.11 recommended for audiocraft)
uv venv --python 3.11
.venv\Scripts\activate

# Install PyTorch with CUDA first (match your CUDA version)
# For CUDA 12.x (RTX 5090):
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
uv pip install -r requirements.txt

# Verify
python -c "import torch; print(f'torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
python -c "from audiocraft.models import AudioGen; print('AudioCraft OK')"
```

> **Note:** If `audiocraft` fails to install via pip, install from source:
> ```powershell
> uv pip install git+https://github.com/facebookresearch/audiocraft.git
> ```

> **Note:** audiocraft requires Python ≤ 3.11. It may not work on 3.12+.

---

## Step 2: Copy Data

Copy the `data/` folder from the other machine. You need at minimum:
- `data/processed/` — all 325 preprocessed .wav files (16kHz mono, 4s each)
- `data/manifest.csv` — metadata with columns: `file_name`, `caption`, `split`

If you don't have the data, regenerate it:
```powershell
python scripts/fetch_minecraft_assets.py --config configs/demo1.yaml
python scripts/preprocess_audio.py --config configs/demo1.yaml
python scripts/build_manifest.py --config configs/demo1.yaml
```

---

## Step 3: Run the Pipeline

### Option A: Run the notebook (recommended)

Open `notebooks/audiogen_finetune.ipynb` in VS Code and run cells in order.
The notebook has 8 sections — see comments in each cell for what to skip.

### Option B: Run from command line

```powershell
# 1. Prepare AudioGen dataset format (JSONL + JSON sidecars)
python scripts/prepare_audiogen_data.py --config configs/demo1.yaml

# 2. Smoke test (10 epochs — verify loss decreases, no NaN)
python -m src.mcaudio.train.audiogen_lora_train --config configs/demo1.yaml --epochs 10 --batch_size 4

# 3. Full training (150 epochs — ~30-60 min on RTX 5090)
python -m src.mcaudio.train.audiogen_lora_train --config configs/demo1.yaml --epochs 150 --batch_size 4

# 4. Generate baseline (vanilla AudioGen, no LoRA)
python -m src.mcaudio.infer.audiogen_generate --prompt "minecraft zombie hurt sound effect" --config configs/demo1.yaml --output outputs/audiogen/baseline

# 5. Generate with LoRA
python -m src.mcaudio.infer.audiogen_generate --prompt "minecraft zombie hurt sound effect" --config configs/demo1.yaml --lora_weights outputs/audiogen/lora_weights/best --output outputs/audiogen/lora
```

---

## Key Technical Details

### Architecture
- **Model:** `facebook/audiogen-medium` (1.5B params) — autoregressive transformer generating EnCodec tokens
- **EnCodec:** 4 codebooks × 2048 vocab × 50Hz frame rate. 4s audio = 4 × 200 = 800 tokens
- **Text conditioning:** Frozen T5 encoder → cross-attention into transformer LM
- **LoRA targets:** `out_proj`, `linear1`, `linear2` (NOT q/k/v — they're fused into `in_proj_weight` in AudioCraft's StreamingMultiheadAttention)

### Training
- **LoRA:** rank 128, alpha 256, dropout 0.05 → ~44M trainable params (vs 1.5B total)
- **FP32 only** — AudioCraft has a known NaN bug with `torch.autocast` / mixed precision
- **Critical fix:** `logits.nan_to_num(nan=0.0)` before cross-entropy (delay codebook pattern produces NaN at offset positions)
- **CFG dropout:** 10% of captions replaced with "" during training for classifier-free guidance at inference
- **Loss:** Masked cross-entropy on valid token positions only

### Config
All hyperparameters live in `configs/demo1.yaml` under the `audiogen:` key. CLI args override config values.

### Checkpoints
- Saved to `outputs/audiogen/lora_weights/` with subdirs: `best/`, `final/`, `epoch_NNNN/`
- Each checkpoint is a single `lora_weights.pt` file (~176 MB) containing only LoRA adapter tensors

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `audiocraft` won't install | Use `uv pip install git+https://github.com/facebookresearch/audiocraft.git` |
| Python 3.12+ errors | AudioCraft needs Python ≤ 3.11 |
| CUDA OOM | Reduce `--batch_size` to 2 or 1. The 5090 (32GB) should handle batch 4-8 easily |
| NaN loss | Should not happen (we use FP32 + nan_to_num). If it does, check that audiocraft is ≥1.3.0 |
| `Import "audiocraft.modules.conditioners" could not be resolved` | IDE warning only — works at runtime |
| Loss doesn't decrease | Try reducing `--lr` to 1e-4. Also check that data prep ran correctly (verify JSONL counts) |
| EnCodec decode sounds wrong | Verify audio files are 16kHz mono. Run the EnCodec roundtrip cell in the notebook |

---

## Project Structure (AudioGen-relevant files)

```
GenAI-Minecraft-Sounds/
├── configs/
│   └── demo1.yaml                         # audiogen: config block at bottom
├── data/
│   ├── manifest.csv                       # 325 clips (file_name, caption, split)
│   ├── processed/                         # 16kHz mono .wav files
│   │   ├── mob/zombie/hurt_seq.wav
│   │   ├── ambient/cave/cave1.wav
│   │   ├── step/stone_walk.wav
│   │   └── ...
│   └── audiogen/                          # created by prepare_audiogen_data.py
│       ├── train.jsonl
│       └── val.jsonl
├── scripts/
│   └── prepare_audiogen_data.py           # manifest.csv → JSONL + sidecars
├── src/mcaudio/
│   ├── train/
│   │   ├── lora_train.py                  # OLD: AudioLDM2 approach (keep for reference)
│   │   └── audiogen_lora_train.py         # NEW: AudioGen LoRA training
│   └── infer/
│       ├── generate.py                    # OLD: AudioLDM2 inference
│       └── audiogen_generate.py           # NEW: AudioGen inference
├── notebooks/
│   ├── demo1_colab.ipynb                  # OLD: AudioLDM2 Colab notebook
│   └── audiogen_finetune.ipynb            # NEW: AudioGen notebook (local GPU)
├── outputs/audiogen/                      # generated audio + checkpoints
│   ├── baseline/
│   ├── lora/
│   └── lora_weights/
│       ├── best/lora_weights.pt
│       ├── final/lora_weights.pt
│       └── epoch_0025/lora_weights.pt
├── docs/
│   └── approaches.md                      # comparison of 5 alternative approaches
├── requirements.txt
└── SETUP_LAB.md                           # ← this file
```
