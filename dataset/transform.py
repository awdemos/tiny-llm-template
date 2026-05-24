def transform(text: str) -> str:
    """Apply a domain-specific text transform to each pretraining example.

    Override this function to customize your corpus (e.g., style transfer,
    augmentation, terminology injection, etc.). The default is a no-op pass-through.
    """
    return text
