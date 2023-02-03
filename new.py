from reflow.data.utils import data2lmdb, get_image_transforms, LMDB_ndarray
from reflow.data.reflow_with_text import DataPairsWithTextLMDB, get_reflow_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
import torch

if __name__ == "__main__":

    # data2lmdb('data/coco2014_reflow/train5M/part4/content/images')
    
    # ds = get_reflow_dataset(
    #     data_root='data/coco2014_reflow/train5M',
    #     src_type='lmdb'
    # )
    # dl = DataLoader(ds, batch_size=4, num_workers=1, shuffle=True)
    # for batch in tqdm(dl):
    #     ...
    
    s = "<s> A bird is sitting on a bowl of birdseed.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>"
    s=s[4:s.find("<pad>")-4]
    print(s)