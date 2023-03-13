import glob
import os
import numpy as np
import pickle
import cv2
import tqdm
import torch
from utils.network_utils import homo2offsets

src = '/media/efklidis/4TB/dblab_real'

# general
train_fr_nr = len(glob.glob(os.path.join(src, 'train', '*', 'GT', '*.png')))
val_fr_nr = len(glob.glob(os.path.join(src, 'val', '*', 'GT', '*.png')))

print(train_fr_nr)
print(val_fr_nr)



# homography
hm_train = sorted(glob.glob(os.path.join(src, 'train', '*', '*.pkl')))
i = homo2offsets(torch.eye(3).unsqueeze(0).cuda(), 800, 800)
mace_train = []
for homo in hm_train:
    with open(homo, 'rb') as homo_pickle:
        homographies = torch.FloatTensor(pickle.load(homo_pickle))
    for homography in homographies:
        off = homo2offsets(homography.unsqueeze(0).cuda(), 800, 800)
        mace_train.append(((off - i) ** 2).sum(dim=2).sqrt().mean())
print(sum(mace_train).item() / len(mace_train))


hm_val = sorted(glob.glob(os.path.join(src, 'val', '*', '*.pkl')))
i = homo2offsets(torch.eye(3).unsqueeze(0).cuda(), 800, 800)
mace_val = []
for homo in hm_val:
    with open(homo, 'rb') as homo_pickle:
        homographies = torch.FloatTensor(pickle.load(homo_pickle))
    for homography in homographies:
        off = homo2offsets(homography.unsqueeze(0).cuda(), 800, 800)
        mace_val.append(((off - i) ** 2).sum(dim=2).sqrt().mean())
print(sum(mace_val).item() / len(mace_val))

# segmentation

masks_train = sorted(glob.glob(os.path.join(src, 'train', '*', 'masks', '*.png')))
area_train = []
for mask in tqdm.tqdm(masks_train):
    area_train.append((cv2.imread(mask) / 255.0).sum() / 800 ** 2)
print(1 -sum(area_train) / len(area_train))

masks_val = sorted(glob.glob(os.path.join(src, 'val', '*', 'masks', '*.png')))
area_val = []
for mask in tqdm.tqdm(masks_val):
    area_val.append((cv2.imread(mask) / 255.0).sum() / 800 ** 2)
print(1 - sum(area_val) / len(area_val))


# restoration
gt_train = sorted(glob.glob(os.path.join(src, 'train', '*', 'GT', '*.png')))
in_train = sorted(glob.glob(os.path.join(src, 'train', '*', 'input', '*.jpg')))
psnr_train = []
for _gt, _in in tqdm.tqdm(zip(gt_train, in_train)):
    psnr_train.append(cv2.PSNR(cv2.imread(_gt), cv2.imread(_in)))
print(sum(psnr_train)/len(psnr_train))

gt_val = sorted(glob.glob(os.path.join(src, 'val', '*', 'GT', '*.png')))
in_val = sorted(glob.glob(os.path.join(src, 'val', '*', 'input', '*.jpg')))
psnr_val = []
for _gt, _in in tqdm.tqdm(zip(gt_val, in_val)):
    psnr_val.append(cv2.PSNR(cv2.imread(_gt), cv2.imread(_in)))
print(sum(psnr_val)/len(psnr_val))