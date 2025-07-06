import os
import os.path as op
import glob
import lmdb
import yaml

from lmdb_process import make_lmdb_from_png
import pdb



def create_lmdb_for_seg():
    root_dir = '../data/lmdbs/'

    img_folder = '../data/imgs_gen/'
    msk_folder = '../data/masks_gen/'
    img_path = 'imgs_gen.lmdb'
    msk_path = 'msks_gen.lmdb'

    lmdb_img_dir = op.join(root_dir, img_path)
    lmdb_msk_dir = op.join(root_dir, msk_path)

    print('scaning pngs...')

    # pdb.set_trace()

    img_list = sorted(os.listdir(img_folder))

    print(f'>{len(img_list)} pngs found.')
    
    print('Writing LMDB for Images data...')
    
    # pdb.set_trace()

    make_lmdb_from_png(img_dir= img_folder, 
                       lmdb_path = lmdb_img_dir,  
                       keys = img_list, 
                       multiprocessing_read=True)
        
    print('Writing LMDB for Masks data...')
    pdb.set_trace()
    make_lmdb_from_png(img_dir= msk_folder, 
                       lmdb_path = lmdb_msk_dir,  
                       keys = img_list, 
                       multiprocessing_read=True)
    
    print("> Finish.")

    # if not op.exists()



if __name__ == "__main__":
    create_lmdb_for_seg()
