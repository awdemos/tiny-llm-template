---
language: en
tags:
  - gpt
  - nanogpt
  - tinystories
  - text-generation
pipeline_tag: text-generation
library_name: pytorch
---

# tiny-llm-template

A minimal, end-to-end template for training tiny domain-specific language models from scratch. Built on the nanoGPT architecture, it gives you a complete two-stage pipeline: pretraining on raw text, then supervised fine-tuning (SFT) to follow instructions. Fork it, swap in your data, and train your own small LM.

## What this is

This repo is a starting point, not a finished product. It provides the scaffolding (model, tokenizer, training loops, data loaders, checkpointing) while you supply the domain knowledge through dataset selection and custom transforms. The default setup trains a ~14M parameter GPT on TinyStories, but you can point it at any text corpus.

The pipeline has two stages:

1. **Pretraining** — trains a base model on raw text with next-token prediction
2. **SFT** — fine-tunes the base model on instruction/response pairs

## Model details

| Field | Value |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Parameters | ~13.9M |
| Layers | 6 |
| Heads | 6 |
| Embedding dim | 384 |
| Context length | 256 tokens |
| Vocab size | 8192 (custom BPE) |
| Bias in Linear/LN | False |
| Tokenizer | `tokenizer.json` (custom BPE) |

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Configure your datasets
# Edit training/config.py and set:
#   pretrain_dataset = "roneneldan/TinyStories"
#   sft_dataset = "databricks/dolly-15k"

# 3. (Optional) Implement a custom transform
# Edit dataset/transform.py to modify each pretraining example

# 4. Run the full data pipeline
make dataset

# 5. Pretrain the base model
make train-gpu

# 6. Fine-tune with SFT
make sft-gpu

# 7. Generate text
make sample PROMPT='Once upon a time'
```

## Usage

This is **not** a `transformers` model. It uses a custom `GPT` class defined in this repo. To load a trained model from the Hugging Face Hub:

```python
import json, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from tokenizers import Tokenizer

from training.config import Config
from training.model import GPT

repo = "your-username/your-model-name"
cfg_path = hf_hub_download(repo, "config.json")
weights_path = hf_hub_download(repo, "model.safetensors")
tok_path = hf_hub_download(repo, "tokenizer.json")

cfg_dict = json.load(open(cfg_path))
cfg = Config(**{k: v for k, v in cfg_dict.items()
                if k in Config.__dataclass_fields__})
model = GPT(cfg).eval()
load_model(model, weights_path)

tok = Tokenizer.from_file(tok_path)
ids = torch.tensor([tok.encode("Once upon a time").ids])
with torch.no_grad():
    for _ in range(80):
        logits, _ = model(ids[:, -cfg.block_size:])
        next_id = torch.multinomial(torch.softmax(logits[:, -1] / 0.8, -1), 1)
        ids = torch.cat([ids, next_id], dim=1)
print(tok.decode(ids[0].tolist()))
```

## Customization

### Swap datasets

Edit `training/config.py` and set `pretrain_dataset` and `sft_dataset` to any Hugging Face dataset identifiers. The pretraining dataset should contain a text field. The SFT dataset should contain `instruction`, `context`, and `response` fields.

### Implement a custom transform

Edit `dataset/transform.py`. The `transform(text: str) -> str` function is applied to every pretraining example before tokenization. Use it for style transfer, content filtering, domain terminology injection, or any other preprocessing.

### Adjust model size

Edit the architecture fields in `training/config.py`:

| Field | What it controls |
|---|---|
| `n_layer` | Number of transformer blocks |
| `n_head` | Attention heads per block |
| `n_embd` | Hidden dimension (must divide evenly by `n_head`) |
| `block_size` | Context length in tokens |
| `vocab_size` | Must match your trained tokenizer |

### Publish to Hugging Face Hub

Set `hf_repo_id` in your config (e.g., `"your-username/your-model-name"`). The training script will push checkpoints automatically. You can also run `python scripts/publish_hf.py` to upload manually.

## Project structure

```
.
├── dataset/          # Data pipeline: download, transform, tokenize
├── training/         # Model, training loops, config, sampling
├── scripts/          # Publishing helpers for HF Hub and Spaces
├── space/            # Gradio app for HF Space deployment
├── Makefile          # Common commands (train, sample, clean)
└── training/config.py # Central configuration for all hyperparameters
```

## Limitations

- Small scale by design. Models trained with default configs are educational artifacts, not competitive with modern LMs.
- Short context window (256 tokens with default config).
- No built-in safety filtering or RLHF. Outputs depend entirely on your training data.
- Best suited for prototyping, learning, and narrow domain tasks.

## License

MIT
