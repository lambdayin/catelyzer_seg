import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
import lmdb
from tqdm import tqdm
import pdb
import cv2


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

class SelfDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        h, w, c = pil_img.shape
        newW, newH = int(scale * w), int(scale * h)     # 目标宽高，执行resize
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = cv2.resize(pil_img, (newW, newH), 0, 0, cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC)

        img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB)

        # pdb.set_trace()

        if is_mask:     # mask二值数据整理，类别超过2的用指定类别表示
            mask = np.zeros((newH, newW), dtype=np.int64)       # 全空数据
            for i, v in enumerate(mask_values):                 #
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:           # single channel
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))  # chw

            if (img > 1).any():
                img = img / 255.0               # norm

            return img

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        
        # pdb.set_trace()
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        
        self.mask_values = [0, 1]
        logging.info(f'Unique mask values: {self.mask_values}')
        # pdb.set_trace()

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)     # 目标宽高，执行resize
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        # pdb.set_trace()

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))  # chw

            if (img > 1).any():
                img = img / 255.0               # norm

            return img
        
    @staticmethod
    def preprocess_noscale(mask_values, pil_img, scale, is_mask):
        img = np.asarray(pil_img)

        # pdb.set_trace()

        if is_mask:
            return img

        else:
            img = img.transpose((2, 0, 1))  # chw

            img = img / 255.0               # norm

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        if self.scale == 1:
            img = self.preprocess_noscale(self.mask_values, img, self.scale, is_mask=False)
            mask = self.preprocess_noscale(self.mask_values, mask, self.scale, is_mask=True)

        else:
            img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
            mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
        # pdb.set_trace()



class LMDBDataset(Dataset):
    def __init__(self, lmdb_path: str, image_lmdb: str, mask_lmdb: str, scale: float = 1.0):
        
        # pdb.set_trace()

        # pdb.set_trace()
        self.images_dir = join(lmdb_path, image_lmdb)
        self.mask_dir = join(lmdb_path, mask_lmdb)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.meta_info_path = join(self.images_dir, "meta_info.txt")

        with open(self.meta_info_path, 'r') as fin:
            self.ids = [line.split(' ')[0] for line in fin]     # get every file name

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.io_opt_dict = dict()
        self.io_opt_dict['type'] = 'lmdb'
        self.io_opt_dict['db_paths']=[self.images_dir, self.mask_dir]
        self.io_opt_dict['client_keys'] = ['img', 'msk']

        self.client = {}
        for client, path in zip(self.io_opt_dict['client_keys'], self.io_opt_dict['db_paths']):
            self.client[client] = lmdb.open(path,
                                            readonly=True,
                                            lock=False,
                                            readahead=False)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(img, scale, is_mask):
        if is_mask:
            h, w = img.shape
        else:
            h, w, c = img.shape
        #  pdb.set_trace()
        if not scale == 1:
            newW, newH = int(scale * w), int(scale * h)     # 目标宽高，执行resize
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            img = cv2.resize(img, (newW, newH), 0, 0, interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC)


        # pdb.set_trace()

        if is_mask:
            return img.astype(np.int64)

        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1))  # chw

            if (img > 1).any():
                img = img / 255.0               # norm

            return img
        

    def __getitem__(self, idx):
        key = self.ids[idx]
        
        client_img = self.client['img']
        with client_img.begin(write=False) as txn:
            img_bytes = txn.get(key.encode('ascii'))

        client_msk = self.client['msk']
        with client_msk.begin(write=False) as txn:
            msk_bytes = txn.get(key.encode('ascii'))
        
        # pdb.set_trace()

        img = _bytes2img(img_bytes)
        msk = _bytes2img(msk_bytes)
        
        img = self.preprocess(img, self.scale, is_mask=False)
        msk = self.preprocess(msk, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(msk.copy()).long().contiguous()
        }


def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    return img
    
class CPUPrefetcher():
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)