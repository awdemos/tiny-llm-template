# tiny-llm-template

## OVERVIEW

Minimal, end-to-end template for training tiny domain-specific language models from scratch. Built on the nanoGPT architecture, it provides a two-stage pipeline: pretraining on raw text, then supervised fine-tuning (SFT). Default model is ~14M parameters.

## STRUCTURE

```
dataset/          Data pipeline: download, transform, tokenize
training/         Model definition, training loops, config, sampling
scripts/          Hugging Face Hub / Space publishing helpers
space/            Gradio app for HF Space deployment
Makefile          Common commands
pyproject.toml    Python project + dependencies
main.py           Placeholder entrypoint
```

## COMMANDS

```bash
make install          # Install Python deps via uv
make env              # Copy example.env -> .env
make dataset          # Full data pipeline: transform + tokenizer + tokenize
make train            # M1 smoke test (CPU/MPS, ~50 iters)
make train-gpu        # Full GPU pretraining (CUDA, bf16)
make sft-gpu          # SFT on configured instruction dataset
make sample PROMPT='Once upon a time'   # Generate from out/ckpt.pt
make clean            # Remove caches and wandb dir
```

Python modules can also be run directly:

```bash
uv run python -m dataset.tiny_stories
uv run python -m training.train
uv run python -m training.sample --prompt "Hello"
```

## SETUP

- Python 3.12+.
- Install `uv` if not present.
- Run `uv sync` or `make install` to install dependencies.
- Run `make env` and fill in Hugging Face / Weights & Biases tokens if needed.
- Configure datasets and model size in `training/config.py`.

## CODE STYLE

- Python with `requires-python = ">=3.12"`.
- Use `make help` to discover available targets.
- Keep the data pipeline in `dataset/` separate from training logic in `training/`.

## DEPLOYMENT

No Dagger module or recognized deployment configuration was found. Models are published to the Hugging Face Hub via `scripts/publish_hf.py`; the Gradio demo in `space/` can be deployed to a Hugging Face Space.
