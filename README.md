# PyExplain: Python Code Bug Explainer

A mini project that combines deterministic Python analysis with an optional QLoRA-fine-tuned open-source LLM. Paste code into the Gradio UI and get syntax errors, likely undefined names, basic safety warnings, and repair guidance.

## Project flow

```text
[data/debugging.jsonl] -> [prepare_dataset.py] -> [train.py / QLoRA adapter] -> [app.py / Gradio]
```

## Setup

Use Python 3.10 or newer. A CUDA-capable GPU is recommended for training.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the app immediately with the built-in static checker:

```powershell
python app.py
```

Open the local URL printed by Gradio. The app works without downloading model weights.

## Fine-tuning route

1. Download and normalize a dataset:

```powershell
python prepare_dataset.py --limit 5000
```

For a first smoke test, use the included sample data:

```powershell
python train.py --data data/sample_debugging.jsonl --epochs 1
```

2. Train the adapter on a CUDA machine:

```powershell
python train.py --data data/debugging.jsonl --epochs 1
```

3. Point the UI at the result and launch it:

```powershell
$env:DEBUGGER_MODEL_PATH = "outputs/code-debugger"
python app.py
```

The model path can also be set in the environment before running `app.py`. The static analyzer always runs first, so its diagnostics remain available when model loading fails.

## Notes

- `prepare_dataset.py` normalizes several common Hugging Face column names into the stable fields `instruction`, `input`, and `output`.
- QLoRA needs a compatible CUDA/PyTorch setup. On CPU or Windows without a working `bitsandbytes` install, use the static checker or train in WSL/Linux/cloud GPU.
- This tool provides suggestions, not a sandbox. Do not execute pasted code inside the app.
