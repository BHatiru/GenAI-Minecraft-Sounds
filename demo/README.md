# Minecraft AudioGen — Live Demo

Interactive Streamlit app comparing **vanilla AudioGen** vs our **LoRA fine-tuned** model side by side.

## Run

```powershell
& ".venv\Scripts\python.exe" -m streamlit run demo/streamlit_app.py
```

Then open <http://localhost:8501>.

## Modes

- **Showcase (instant)** — plays pre-generated curated samples from `outputs/audiogen/{baseline,lora,generalization}/`. Two tabs:
  - *In-Domain Prompts* — 4 standard Minecraft sound categories
  - *Generalization Prompts* — 10 unseen multi-event prompts (4–12 s)
- **Live generation** — loads both AudioGen models into VRAM and generates on demand. Requires ~6 GB VRAM. First load ~30 s; each generation 10–30 s on RTX 5090.

## What it shows

For each prompt, the app renders a side-by-side comparison:
- 🟦 **Baseline AudioGen** (`facebook/audiogen-medium`, vanilla)
- 🟪 **LoRA Fine-Tuned** (rank=128, α=256, ~132 M trainable params)

Each panel includes an audio player and a log-power spectrogram so the audience can both hear and see the difference.

## Sidebar metrics

| | Baseline | LoRA | Δ |
|---|---:|---:|---:|
| FAD ↓ | 6171 | **3856** | **−37 %** |
| KAD ↓ | 9.1 e8 | **2.5 e8** | **−73 %** |
| CLAP ↑ | 0.142 | **0.162** | +14 % |

## Files

- `streamlit_app.py` — main app
- LoRA weights loaded from `outputs/audiogen/lora_weights/best/`
- Live generations saved to `outputs/audiogen/demo_live/`
