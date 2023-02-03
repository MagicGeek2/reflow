from torchvision import transforms as T
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pickle
import lmdb
from pathlib import Path


def array_to_tensor(array):
    return torch.from_numpy(array)

def get_image_transforms(train=False, random_flip=False):
    image_transforms=[array_to_tensor]
    if train:
        if random_flip:
            image_transforms.append(T.RandomHorizontalFlip())
    else:
        ...
    image_transforms = T.Compose(image_transforms)
    return image_transforms


# * 用于从大量 small npy 文件中创建 lmdb 
class LMDB_ndarray:
    def __init__(self, array: np.ndarray):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.shape = array.shape
        self.dtype = array.dtype
        self.array = array.tobytes()

    def resume_array(self):
        """ Returns the image as a numpy array. """
        array = np.frombuffer(
            self.array, dtype=self.dtype).reshape(*self.shape)
        return array
    
class NumpyPaths(Dataset):
    def __init__(self, data_root) -> None:
        super().__init__()
        self.paths = glob(f'{data_root}/*.npy')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        array = np.load(self.paths[i])
        return array
    
def data2lmdb(dpath, write_frequency=5000, num_workers=8):
    # 自定义数据集
    ds = NumpyPaths(dpath)
    dl = DataLoader(ds, num_workers=num_workers, collate_fn=lambda x: x)

    dpath = Path(dpath)
    lmdb_path = dpath.parent / f'lmdb'
    lmdb_path.mkdir(parents=True, exist_ok=True)

    print(f"Generate LMDB to {str(lmdb_path)}")
    db = lmdb.open(str(lmdb_path), subdir=lmdb_path.is_dir(),
                   map_size=1099511627776, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    for idx, data in enumerate(dl):
        array = data[0]
        temp = LMDB_ndarray(array)
        txn.put(f'{idx}'.encode('utf-8'), pickle.dumps(temp))
        if (idx+1) % write_frequency == 0:
            print(f"[{idx+1}/{len(dl)}]")
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [f'{k}'.encode('utf-8') for k in range(len(dl))]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))

    print("Flushing database...")
    db.sync()
    db.close()
    print('done!')
    
if __name__ == "__main__":
    
    # * 正式生成数据
    data2lmdb('data/coco2014_reflow/train5M/part1/content/images', write_frequency=5000, num_workers=8)
    