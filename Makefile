.PHONY: help install env data tokenizer tokens dataset train train-gpu sft-gpu sample clean clean-data clean-ckpt

UV ?= uv
PROMPT ?= Once upon a time

help:
	@echo "tiny-llm-template make targets"
	@echo ""
	@echo "  make install       Install Python deps via uv"
	@echo "  make env           Copy example.env -> .env (won't overwrite existing)"
	@echo ""
	@echo "  make dataset       Full data pipeline: transform + tokenizer + tokenize"
	@echo "  make data          Download and transform pretraining dataset -> dataset/processed"
	@echo "  make tokenizer     Train BPE tokenizer  -> tokenizer.json"
	@echo "  make tokens        Tokenize corpus      -> train.bin, val.bin"
	@echo ""
	@echo "  make train         M1 smoke test (CPU/MPS, ~50 iters)"
	@echo "  make train-gpu     Full GPU run (CUDA, bf16)"
	@echo "  make sft-gpu       SFT on configured instruction dataset (CUDA, loads pretrained ckpt)"
	@echo ""
	@echo "  make sample PROMPT='Once upon a time'   Generate from out/ckpt.pt"
	@echo ""
	@echo "  make clean         Remove caches + wandb dir"
	@echo "  make clean-ckpt    Remove out/ckpt.pt"
	@echo "  make clean-data    Remove tokenized bins + tokenizer + processed dataset"

install:
	$(UV) sync

env:
	@if [ -f .env ]; then \
		echo ".env already exists — leaving alone"; \
	else \
		cp example.env .env && echo "Created .env from example.env — fill in your tokens"; \
	fi

# ----- Data pipeline -----

dataset/processed:
	$(UV) run python -m dataset.tiny_stories

data: dataset/processed
pretrain-data: dataset/processed

tokenizer.json: dataset/processed
	$(UV) run python -m dataset.tokenize_ds

tokenizer: tokenizer.json

train.bin val.bin: tokenizer.json dataset/processed
	$(UV) run python -m dataset.tokenize_corpus

tokens: train.bin val.bin

dataset: data tokenizer tokens

# ----- Training -----

train: train.bin val.bin
	$(UV) run python -m training.train

train-gpu: train.bin val.bin
	$(UV) run python -c "from training.config import Config; from training.train import train; train(Config.for_gpu_training())"

sft-gpu:
	$(UV) run python -m training.sft_train

# ----- Sampling -----

sample:
	$(UV) run python -m training.sample --prompt "$(PROMPT)"

# ----- Cleanup -----

clean:
	rm -rf wandb/ .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-ckpt:
	rm -rf out/

clean-data:
	rm -f train.bin val.bin tokenizer.json
	rm -rf dataset/processed
