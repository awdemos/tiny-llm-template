from datasets import DatasetDict, load_dataset

from dataset.transform import transform
from training.config import Config

if __name__ == "__main__":
    config = Config()
    if not config.pretrain_dataset:
        raise ValueError("pretrain_dataset must be set in training/config.py")

    raw_ds = load_dataset(config.pretrain_dataset)
    processed_ds = DatasetDict(
        {k: v.map(lambda x: {"text": transform(x["text"])}) for k, v in raw_ds.items()}
    )
    processed_ds.save_to_disk("dataset/processed")
