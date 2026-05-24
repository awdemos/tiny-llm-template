# ADR 0002: Mandatory Dataset Configuration

**Status:** Accepted

**Date:** 2026-05-24

---

## Context

The template supports two distinct training stages, each requiring a dataset: pretraining (raw text) and SFT (prompt/response pairs). We needed a policy for how users specify which datasets to use.

We considered two approaches:

1. **Mandatory explicit config:** Users must set `pretrain_dataset` and `sft_dataset` in `config.py` before running either pipeline. There are no hardcoded defaults. If a field is missing or None, the pipeline raises a clear error explaining what to configure.
2. **Default datasets with override:** Ship the template with default datasets (e.g., TinyStories for pretraining, a small instruction set for SFT) already filled in. Users can override them if they want something else, but the template runs out of the box without any edits.

## Decision

We will **require users to explicitly configure both datasets** in `config.py`. There are no hardcoded default dataset identifiers.

## Consequences

**Positive:**

- **User intentionality.** A user cannot accidentally train on the wrong dataset because they forgot to override a default. Setting the dataset is an explicit act that forces the user to think about their data source.
- **No surprise downloads.** Default datasets can trigger large, unexpected downloads on the first run. Mandatory config means the user knows exactly what they are fetching.
- **Clear error messages.** When a required field is missing, the pipeline can tell the user exactly which config key to set and what format it expects. This is easier to debug than "why is my model speaking like TinyStories?"
- **Domain alignment.** This template is meant to be forked for specific domains. A user building a medical-question-answering model should not have to remember to remove a default story dataset. Starting from a blank slate matches the fork-and-customize workflow.

**Negative:**

- **Higher initial friction.** A user cannot clone the repo and run a training pipeline immediately. They must first choose a dataset, understand its format, and enter it into `config.py`.
- **Steeper learning curve for beginners.** Someone who just wants to see the training loop execute must now figure out where to find a compatible dataset and how to specify it.
- **No instant smoke test.** The default-dataset approach provides a built-in end-to-end test: clone, run, verify it works. With mandatory config, the smoke test requires an extra setup step.

## Rationale

The template is designed as a forkable starting point for domain-specific tiny LLMs, not as a runnable demo. In that context, training on the wrong data is a more serious failure mode than needing five extra minutes of setup. Mandatory config protects users from accidentally baking a default dataset into their model weights.

To mitigate the friction, the `config.py` file includes commented examples showing the expected format for both `pretrain_dataset` and `sft_dataset`. The error messages raised by the pipelines also point directly to those examples. The extra setup step is a small price for eliminating an entire class of silent misconfiguration bugs.
