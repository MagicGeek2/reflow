import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import numpy as np
from loguru import logger


def tokenize_caption(caption, tokenizer):
    tokens_pt = tokenizer(
        caption,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    for k,v in tokens_pt.items():
        tokens_pt[k]=v.squeeze()
    return tokens_pt


class DataPairsWithText(Dataset):
    def __init__(self, data_root, phase, image_transforms=None, tokenizer=None) -> None:
        super().__init__()
        data_root = Path(data_root)
        assert phase in ['train', 'val']
        data_root = data_root / phase
        self.index_info = json.load(open(str(data_root / 'index.json'), 'r'))
        logger.info(f'dataset basic info:\n{self.index_info}')

        content_dir = data_root / 'content'
        self.image_dir = content_dir / 'images'
        self.all_caps = open(
            str(content_dir / 'captions.txt'), 'r').read().splitlines()
        self.nums = len(self.all_caps)

        self.image_transforms = image_transforms
        self.tokenizer = tokenizer

    def __len__(self):
        return self.nums

    def __getitem__(self, i):
        example = {}
        pair = np.load(str(self.image_dir / f'{i}.npy'))
        noise, latent = pair  # 2 * (d, h, w) np.ndarray
        if self.image_transforms:
            example['noise'] = self.image_transforms(noise)
            example['latent'] = self.image_transforms(latent)
        assert self.tokenizer
        caption = self.all_caps[i]
        tokens_pt = tokenize_caption(caption, self.tokenizer)
        example = {**example, **tokens_pt}
        return example

    def add_transforms_(self, image_transforms=None, tokenizer=None):
        if image_transforms:
            self.image_transforms = image_transforms
        if tokenizer:
            self.tokenizer = tokenizer


def collate_fn(examples, tokenizer):
    noise = torch.stack([example["noise"] for example in examples])
    noise = noise.to(memory_format=torch.contiguous_format).float()
    latent = torch.stack([example["latent"] for example in examples])
    latent = latent.to(memory_format=torch.contiguous_format).float()

    input_ids = [example["input_ids"] for example in examples]
    padded_tokens = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt")

    return {
        "noise": noise,
        "latent": latent,
        "input_ids": padded_tokens.input_ids,
        "attention_mask": padded_tokens.attention_mask,
    }
