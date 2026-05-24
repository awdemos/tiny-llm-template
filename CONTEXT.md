# Context

This glossary defines the core concepts and terminology used throughout the tiny-llm-template project.

---

## Transform

A user-defined text transformation applied to each example in the pretraining corpus. The canonical location is `dataset/transform.py`, which exports a standard `transform` function that takes a raw example and returns the modified text. By default, the transform is a no-op (it returns the input unchanged). This hook exists so users can inject domain-specific preprocessing — for example, rewriting text to match a particular style, stripping unwanted content, or normalizing formatting — before the corpus is tokenized and fed into the model.

## Pretraining Pipeline

The first training stage, where the model learns language structure from raw text. The pipeline follows these steps in order: download a source dataset from Hugging Face, optionally apply a user-defined transform, train a custom BPE tokenizer on the transformed corpus, tokenize the full dataset into binary files for fast random access, and train the GPT model on those binary files using next-token prediction. The output of this stage is a base model with general language understanding.

## SFT (Supervised Fine-Tuning)

The second training stage, where the pretrained model learns to follow instructions. SFT uses prompt/response pairs drawn from a dataset with three standard fields: `instruction` (what the user wants), `context` (optional background information), and `response` (the desired output). During training, loss is computed only on the response tokens, not on the instruction or context. This teaches the model to generate helpful, relevant answers rather than simply continuing arbitrary text.

## Template

This repository itself, treated as a starting point rather than a finished product. Users fork the template, configure their own datasets and hyperparameters in `config.py`, implement any custom transform logic in `dataset/transform.py`, and run the pretraining and SFT pipelines to produce a domain-specific tiny LLM. The template provides the scaffolding — model architecture, training loops, data loaders, and checkpointing — while the user supplies the domain knowledge through data selection and transforms.

## Config

The central configuration object, defined as a dataclass in `training/config.py`. It holds every hyperparameter that controls training behavior: model architecture dimensions (layers, heads, embedding size), optimizer settings (learning rate, weight decay, gradient clipping), training schedule (warmup steps, total iterations, evaluation frequency), hardware choices (device, dtype, compilation), and project identity (run name, output directory). The Config is the single source of truth for a training run. Two factory methods, `for_m1_smoke_test` and `for_gpu_training`, provide sensible starting points for common hardware profiles.
