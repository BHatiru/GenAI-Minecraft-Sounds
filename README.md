# GenAI-Minecraft-Sounds

Train a diffusion audio model (AudioLDM2) to generate Minecraft-style sound effects via LoRA adaptation.

## Project Structure

```
GenAI-Minecraft-Sounds/
├── configs/
│   └── demo1.yaml              # paths + hyperparams
├── scripts/
│   ├── fetch_minecraft_assets.py   # download .ogg from GitHub
│   ├── preprocess_audio.py         # ogg → 16 kHz mono wav
│   └── build_manifest.py           # metadata.csv + train/val split
├── src/mcaudio/
│   ├── infer/
│   │   └── generate.py            # AudioLDM2 inference
│   └── train/
│       └── lora_train.py          # LoRA fine-tuning entrypoint
├── notebooks/
│   └── demo1_colab.ipynb          # end-to-end Colab workflow
├── requirements.txt
└── .gitignore                     # excludes data/ and outputs/
```

## Demo 1 Quickstart

### Option A – Google Colab (recommended)

1. Upload the repo to GitHub.
2. Open `notebooks/demo1_colab.ipynb` in Colab (T4 GPU runtime).
3. Update the `REPO_URL` in cell 2.
4. Run cells in order – each stage is self-contained.

### Option B – Local / CLI

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Minecraft sound assets
python scripts/fetch_minecraft_assets.py --config configs/demo1.yaml

# 3. Preprocess audio (ogg → 16 kHz mono wav, 4 s fixed length)
python scripts/preprocess_audio.py --config configs/demo1.yaml

# 4. Build manifest (metadata.csv with captions + train/val split)
python scripts/build_manifest.py --config configs/demo1.yaml

# 5. Generate baseline samples (vanilla AudioLDM2)
python -m src.mcaudio.infer.generate \
    --prompt "minecraft zombie hurt sound effect" \
    --config configs/demo1.yaml \
    --num_samples 4

# 6. (Optional) LoRA fine-tune
python -m src.mcaudio.train.lora_train --config configs/demo1.yaml

# 7. (Optional) Generate with LoRA adapter
python -m src.mcaudio.infer.generate \
    --prompt "minecraft zombie hurt sound effect" \
    --config configs/demo1.yaml \
    --lora_weights outputs/demo1/lora_weights \
    --output outputs/demo1/lora
```

### Artefact Locations

| Stage | Output | Path |
|-------|--------|------|
| Raw assets | .ogg files | `data/raw/` |
| Processed | 16 kHz mono .wav | `data/processed/` |
| Manifest | metadata.csv | `data/manifest.csv` |
| Baseline samples | generated .wav | `outputs/demo1/baseline/` |
| LoRA weights | adapter checkpoint | `outputs/demo1/lora_weights/` |
| LoRA samples | generated .wav | `outputs/demo1/lora/` |

### Audio Spec

- Sample rate: 16 kHz
- Channels: mono
- Format: float32 WAV in [-1, 1]
- Fixed length: 4 seconds (padded or clipped)
- Silence trimmed at 30 dB threshold
