import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import json
import numpy as np
from loguru import logger
import lmdb
import pickle
from reflow.data.utils import get_image_transforms
from glob import glob
from reflow.data.utils import LMDB_ndarray



def tokenize_caption(caption, tokenizer=None):
    if tokenizer == None:
        return {
            'caption': caption,
        }

    tokens_pt = tokenizer(
        caption,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    for k, v in tokens_pt.items():
        tokens_pt[k] = v.squeeze()
    return tokens_pt


def get_reflow_dataset(
    data_root,
    tokenizer=None,
    src_type='npy',

    train=False,
    random_flip=False,
):
    assert src_type in ['npy', 'lmdb']
    ds_clsx = {
        'npy': DataPairsWithText,
        'lmdb': DataPairsWithTextLMDB,
    }
    image_transforms = get_image_transforms(
        train=train, random_flip=random_flip)
    ds_cls = ds_clsx[src_type]
    data_root = Path(data_root)
    if (data_root / 'index.json').exists():
        ds = ds_cls(
            data_root=str(data_root),
            image_transforms=image_transforms,
            tokenizer=tokenizer,
        )
    else:
        # 需要加载合并所有的 part
        proot_list = glob(str(data_root / 'part*'))
        logger.info(f'find {len(proot_list)} different parts : {proot_list}')
        ds_list = []
        for proot in proot_list:
            try:
                ds_list.append(ds_cls(
                    data_root=str(proot),
                    image_transforms=image_transforms,
                    tokenizer=tokenizer,
                ))
            except:
                logger.warning(f'{proot} no {src_type}')
        ds=ConcatDataset(ds_list)
    logger.info(f'{len(ds)} items in total')
    return ds


class DataPairsWithTextLMDB(Dataset):
    def __init__(self, data_root, image_transforms=None, tokenizer=None):
        data_root = Path(data_root)
        self.index_info = json.load(open(str(data_root / 'index.json'), 'r'))
        logger.info(f'dataset basic info:\n{self.index_info}')

        content_dir = data_root / 'content'
        self.lmdb_dir = content_dir / 'lmdb'
        self.env = lmdb.open(str(self.lmdb_dir), subdir=self.lmdb_dir.is_dir(),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

        self.all_caps = open(
            str(content_dir / 'captions.txt'), 'r').read().splitlines()
        self.image_transforms = image_transforms
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        env = self.env
        with env.begin() as txn:
            byteflow = txn.get(self.keys[index])
        lmdb_array = pickle.loads(byteflow)
        array = lmdb_array.resume_array().copy().astype(np.float32)

        noise, latent = array  # 2 * (d, h, w) np.ndarray
        example = {}
        if self.image_transforms:
            example['noise'] = self.image_transforms(noise)
            example['latent'] = self.image_transforms(latent)
        # assert self.tokenizer
        caption = self.all_caps[index]
        tokens_pt = tokenize_caption(caption, self.tokenizer)
        example = {**example, **tokens_pt}
        return example

    def __len__(self):
        return self.length


class DataPairsWithText(Dataset):
    def __init__(self, data_root, image_transforms=None, tokenizer=None) -> None:
        super().__init__()
        data_root = Path(data_root)
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
        pair = np.load(str(self.image_dir / f'{i}.npy')).astype(np.float32)
        example = {}
        noise, latent = pair  # 2 * (d, h, w) np.ndarray
        if self.image_transforms:
            example['noise'] = self.image_transforms(noise)
            example['latent'] = self.image_transforms(latent)
        # assert self.tokenizer
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


if __name__ == "__main__":

    ...
