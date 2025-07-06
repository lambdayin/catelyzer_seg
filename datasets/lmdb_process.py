import lmdb
import os.path as op
import cv2
from tqdm import tqdm 
from multiprocessing import Pool

import pdb

def _read_png_worker(path, 
                     key,
                     compress_level=1):
    img = cv2.imread(op.join(path, key), cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        height, width = img.shape
        c = 1
    else:
        height,width,c = img.shape
    _, img_byte, = cv2.imencode(
        '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
    )
    return (key, img_byte, (height,width,c))


def make_lmdb_from_png(img_dir, 
                       lmdb_path,  
                       keys, 
                       compress_level=1,
                       batch=5000,
                       multiprocessing_read=False,
                       map_size=None):
    # pdb.set_trace()
    assert lmdb_path.endswith('.lmdb'), "lmdb_path must end with '.lmdb'."
    assert not op.exists(lmdb_path), f'Folder {lmdb_path} already exists'

    num_img = len(keys)

    assert multiprocessing_read, 'Not implemented.'


    def _callback(arg):
        """Registering imgs and shape into the dict & updata pbar."""
        key, img_byte, img_shape = arg
        dataset[key], shapes[key] = img_byte, img_shape
        pbar.set_description(f'Reading {key}')
        pbar.update(1)

    dataset = {}
    shapes = {}
    pbar = tqdm(total=num_img, ncols=200)
    pool = Pool()   # multi_thread processing

    # read an image, record its byte and shape into dict
    for iter_frm in range(num_img):
        pool.apply_async(
            _read_png_worker,
            args = (img_dir,
                keys[iter_frm],
                compress_level
            ),
            callback=_callback
        )

    pool.close()
    pool.join()
    pbar.close()

    pdb.set_trace()

    # obtain data size of one image
    if map_size is None:
       _, img_byte,_ = _read_png_worker(img_dir,keys[0],compress_level)
       map_size = img_byte.nbytes * num_img * 4

    # create lmdb environment & write data to lmdb
    env = lmdb.open(lmdb_path, map_size = map_size)
    txn = env.begin(write=True)
    txt_file = open(op.join(lmdb_path, 'meta_info.txt'), 'w')
    pbar = tqdm(total=num_img, ncols=200)
    
    for idx,key in enumerate(keys):
        pbar.set_description(f'Writing {key}')
        pbar.update(1)

        # load image bytes
        img_byte = dataset[key]
        h,w,c = shapes[key]

        # Write lmdb
        key_byte = key.encode('ascii')
        txn.put(key_byte, img_byte)

        # Write meta
        txt_file.write(f'{key} ({h},{w},{c}) {compress_level}\n')

        # if idx % batch == 0:
        #     txn.commit()
        #     txn = env.begin(write=True)

    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()



    return 0




