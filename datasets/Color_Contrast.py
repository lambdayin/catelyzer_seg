import cv2
import os
import numpy as np
import random
import pdb

def check_and_split(path):
    files = os.listdir(path)
    input = []
    gt = []

    count = 0
    for file in files:
        count = count + 1
        name = file.split("_")[1]

        if name == "groundtruth":
            gt.append(file)
        else:
            input.append(file)

    input.sort()
    gt.sort()

    pdb.set_trace()


    for i, name in enumerate(input):
        parts = name.split("_")
        rest = name.split(parts[1])[-1]
        gt_name = "_groundtruth_(1)_" + parts[0] + rest

        # pdb.set_trace()

        # check dataset in pairs
        if not gt[i] == gt_name:
            print("wrong name!") 
            continue


        # split to two folders and match name
        src_gen = "_groundtruth_\(1\)_" + parts[0] + rest
        command = "mv data/imgs_all/" + src_gen + " data/masks_all/" + name

        # pdb.set_trace()

        os.system(command)
        print(f"{i + 1} files has been matched!")



def Col_Bri_Con_Aug(imgs_path, msks_path, imgs_dst, msks_dst):

    # pdb.set_trace()
    files = os.listdir(imgs_path)
    
    length = len(files)
    
    for idx, file in enumerate(files):
        src_in = cv2.imread(imgs_path + file, cv2.IMREAD_UNCHANGED)

        cmd_src_img = "cp " + imgs_path + file + " " + imgs_dst + file
        os.system(cmd_src_img)
        cmd_src_msk = "cp " + msks_path + file + " " + msks_dst + file
        os.system(cmd_src_msk)


        src_HLS = cv2.cvtColor(src_in, cv2.COLOR_BGR2HLS_FULL)
        # pdb.set_trace 
        
        Aug_Ratio = 9

        # Brightness
        bri_idx=random.sample(range(0,Aug_Ratio), 3)
        # Contrast
        con_idx=random.sample(range(0,Aug_Ratio), 3)

        hue_str=random.sample(range(0,255), Aug_Ratio)

        # pdb.set_trace()


        # hue adjust
        for i in range(Aug_Ratio):  # 随机改变Hue
            Adjusted_HLS = Adjust_HLS(src_HLS, i in bri_idx, i in con_idx, hue_str[i])
            dst_bgr = cv2.cvtColor(Adjusted_HLS, cv2.COLOR_HLS2BGR_FULL)

            # save datas
            parts = os.path.splitext(file)
            # pdb.set_trace()

            save_name = parts[0] + f"_{i}" + parts[1] 
            cv2.imwrite(imgs_dst + save_name, dst_bgr)
            
            command = "cp " + msks_path + file + " " + msks_dst + save_name
            # pdb.set_trace()

            os.system(command)

        if(idx % 100 == 0):
            cv2.imshow("in", src_in)
            cv2.imshow("Adjusted res", dst_bgr)
            cv2.waitKey(10)
        print(f"{idx + 1} / {length} files have been Augmented! please wait....... ")



def Adjust_HLS(src, BriAdj, ConAdj, HueBias):
    # hue
    dst = src.copy()
    # pdb.set_trace()
    dst[:,:,0] = dst[:,:,0] + HueBias
    # dst[:,:,0]= dst[:,:,0] - (dst[:,:,0]>255) * 255

    # bri
    if BriAdj:
        # pdb.set_trace()
        bri = random.randint(10,50)
        dst[:,:,1] = np.clip(dst[:,:,1].astype(np.uint32) + bri,0,255).astype(np.uint8)

    if ConAdj:
        # pdb.set_trace()
        con = random.uniform(0.8,1.5)
        dst[:,:,1] = np.clip(((dst[:,:,1].astype(np.float32) - 128) * con) + 128, 0, 255).astype(np.uint8)

    return dst


    pdb.set_trace()

if __name__ == "__main__":

    imgs_path = "data/imgs_all/"
    
    # check_and_split(path)

    msks_path = "data/masks_all/"


    imgs_dst = "data/imgs_gen/"
    msks_dst = "data/masks_gen/"
    Col_Bri_Con_Aug(imgs_path, msks_path, imgs_dst, msks_dst)

