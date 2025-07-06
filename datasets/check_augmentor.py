import os
import cv2
import numpy as np

import pdb


path = "data/imgs_192/output/"
files = os.listdir(path)
input = []
gt = []

# pdb.set_trace()

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

# pdb.set_trace()

for i, file in enumerate(input):
    img = cv2.imread(path + file,cv2.IMREAD_UNCHANGED)
    msk = cv2.imread(path + gt[i], cv2.IMREAD_UNCHANGED)

    msk = (msk * 255).astype(np.uint8)

    cv2.imshow('img',img)
    cv2.imshow('msk',msk)

    mm = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
    wei = cv2.addWeighted(img,0.5,mm,0.5,gamma=0)

    cv2.imshow('comb',wei)
    cv2.waitKey()

