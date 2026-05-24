# ADR 0001: Module-Based Convention for Transforms

**Status:** Accepted

**Date:** 2026-05-24

---

## Context

Every user of this template will want to preprocess their pretraining corpus differently. One might want to strip boilerplate, another might want to rewrite text in a specific style, and a third might want to filter out examples that match certain patterns. We needed a convention for how users plug in their custom logic.

We considered three approaches:

1. **Module-based convention:** Users write their transform in `dataset/transform.py` as a standard function. The pipeline imports and calls it directly.
2. **Registry pattern:** Users decorate their transform with `@register_transform(name="my_transform")` and select it by name in `config.py`.
3. **Config-driven dynamic imports:** Users specify a Python import path (e.g., `"my_package.transforms:clean_text"`) in `config.py`, and the pipeline dynamically imports and calls it.

## Decision

We will use a **module-based convention** (`dataset/transform.py` with a standard `transform` function). There is no registry, no decorator, and no dynamic import string in `config.py`.

## Consequences

**Positive:**

- **Simplicity.** A new user can look at `dataset/transform.py`, see exactly one function, and understand immediately what to change. There is no indirection, no registration boilerplate, and no config key to remember.
- **Explicitness.** The transform is always in the same file, with the same name. When reading someone else's fork, you know exactly where to look for the preprocessing logic.
- **Type safety and static analysis.** Because the import is a regular static import, type checkers and IDEs can follow the symbol without ambiguity.
- **No accidental complexity.** A registry or dynamic import system would require documentation, error handling for missing transforms, and validation logic. The module convention avoids all of that.

**Negative:**

- **Less flexibility.** Users cannot switch between multiple transforms at runtime by changing a config value. If they want to compare two transforms, they must swap out the file or use version control.
- **No plugin ecosystem.** There is no clean way to share transforms as reusable packages. A user cannot pip-install a third-party transform and reference it from `config.py`.
- **Slightly more friction for experimentation.** Each transform experiment requires editing a file rather than toggling a string in a config.

## Rationale

This template targets users who are learning how to train small language models from scratch. For that audience, every layer of indirection is a barrier to understanding. The module convention makes the transform mechanism impossible to miss and trivial to modify. The lost flexibility is an acceptable trade-off because the primary goal is clarity, not runtime configurability.

If a power user later needs multiple transforms or dynamic switching, they can easily refactor `dataset/transform.py` into a registry pattern themselves. The template does not prevent that; it simply does not impose it by default.
