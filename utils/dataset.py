# This code is referenced from
# https://github.com/facebookresearch/astmt/
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# License: Attribution-NonCommercial 4.0 International

import os
import pickle

import cv2
import torch.utils.data as data
import numpy as np


class MTL_Dataset(data.Dataset):
    """
    NYUD dataset for multi-task learning.
    Includes edge detection, semantic segmentation, surface normals, and depth prediction
    """

    def __init__(self, tasks,
                 root='./DST-dataset/', split='train', seq_len=20, transform=None,
                 meta=True, overfit=False):

        do_seg = 'segment' in tasks
        do_deblur = 'deblur' in tasks
        do_flow = 'flow' in tasks
        self.root = root
        self.transform = transform
        self.meta = meta
        self.split = split
        self.threshold = 40
        # Images
        self.images = []
        # Semantic Segmentation
        self.do_seg = do_seg
        self.masks = []
        # Deblurring
        self.do_deblur = do_deblur
        self.deblur_frames = []
        # Homographies
        self.do_flow = do_flow
        self.flows = []

        dataset_dir = os.path.join(root, self.split)
        for video in os.listdir(dataset_dir):
            image_dir, deblur_dir, mask_dir, of_dir = [os.path.join(dataset_dir, video, directory)
                                                         for directory in ['input', 'GT', 'masks', 'OF_backw']]

            filenames = sorted(os.listdir(image_dir), key=lambda x: int(os.path.basename(x)[:-4]))
            if seq_len is None:
                seq_len_inner = len(filenames)
            else:
                seq_len_inner = seq_len

            for first_idx in range(0, len(filenames)-seq_len_inner+1, seq_len_inner):

                seq_images, seq_masks, seq_deblur_frames, seq_flows = ([] for _ in range(4))

                for file_idx in range(first_idx, first_idx + seq_len_inner):

                    filename = filenames[file_idx]
                    # Images
                    _image = os.path.join(image_dir, filename)
                    assert os.path.isfile(_image)
                    seq_images.append(_image)

                    # Semantic Segmentation
                    if do_seg:
                        _mask = os.path.join(mask_dir, filename[:-4] + '.png')
                        assert os.path.isfile(_mask)
                        seq_masks.append(_mask)

                    # Deblurring
                    if do_deblur:
                        _deblur_frame = os.path.join(deblur_dir, filename[:-4] + '.png')
                        assert os.path.isfile(_deblur_frame)
                        seq_deblur_frames.append(_deblur_frame)

                    # Flows
                    if do_flow:

                        if file_idx==0:
                            _optical_flow = os.path.join(of_dir, filenames[file_idx+1][:-4] + '.npy')
                            assert os.path.isfile(_optical_flow)
                            seq_flows.append(_optical_flow)

                        else:
                            _optical_flow = os.path.join(of_dir, filename[:-4] + '.npy')
                            assert os.path.isfile(_optical_flow)
                            seq_flows.append(_optical_flow)

                self.images.append(seq_images)
                self.masks.append(seq_masks)
                self.deblur_frames.append(seq_deblur_frames)
                self.flows.append(seq_flows)

        if seq_len is None:
            return
        # Uncomment to overfit to one sequence
        if overfit:
            n_of = seq_len
            self.images = [self.images[0]][:n_of]

        # Display stats
        print('Number of {} dataset sequences: {:d}'.format(self.split, len(self.images)))
        print('Number of sequence frames: {:d}'.format(seq_len))

    def __getitem__(self, index):

        _img = self._load_img(index)
        sample = {'image': _img}

        if self.do_seg:
            sample['segment'] = self._load_mask(index)

        if self.do_deblur:
            sample['deblur'] = self._load_deblur(index)

        if self.do_flow:
            sample['flow'] = self._load_flow(index)

        if self.meta:
            sample['meta'] = {'paths': self.images[index],
                              'im_size': (_img[0].shape[0], _img[0].shape[1]),
                              'transformations': {}}

        if self.transform is not None:
            sample = self.transform(sample)


        return sample

    def _load_img(self, index):
        return [cv2.imread(path).astype(np.float32) for path in self.images[index]]

    def _load_deblur(self, index):
        return [cv2.imread(path).astype(np.float32) for path in self.deblur_frames[index]]

    def _load_mask(self, index):
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) for path in self.masks[index]]

    def _load_flow(self, index):
        return [np.load(path).astype(np.float32) for path in self.flows[index]]


    def __len__(self):
        return len(self.images)

    def __str__(self):
        return 'DST Multitask (split=' + str(self.split) + ')'


class MTL_TestDataset(data.Dataset):
    """
    NYUD dataset for multi-task learning.
    Includes edge detection, semantic segmentation, surface normals, and depth prediction
    """

    def __init__(self, tasks,
                 root='./DST-dataset/', split='train', seq_len=None, transform=None,
                 meta=True, overfit=False):

        do_seg = 'segment' in tasks
        do_deblur = 'deblur' in tasks
        do_flow = 'flow' in tasks
        self.root = root
        self.transform = transform
        self.meta = meta
        self.split = split
        self.threshold = 40
        # Images
        self.images = []
        # Semantic Segmentation
        self.do_seg = do_seg
        self.masks = []
        # Deblurring
        self.do_deblur = do_deblur
        self.deblur_frames = []
        # Homographies
        self.do_flow = do_flow
        self.flows = []

        dataset_dir = os.path.join(root, self.split)
        for video in os.listdir(dataset_dir):
            image_dir, deblur_dir, mask_dir, of_dir = [os.path.join(dataset_dir, video, directory)
                                                         for directory in ['input', 'GT', 'masks', 'OF_backw']]

            filenames = sorted(os.listdir(image_dir), key=lambda x: int(os.path.basename(x)[:-4]))
            if seq_len is None:
                seq_len_inner = len(filenames)
            else:
                seq_len_inner = seq_len
            for first_idx in range(0, len(filenames)-seq_len_inner+1, seq_len_inner):

                seq_images, seq_masks, seq_deblur_frames, seq_flows = ([] for _ in range(4))

                for file_idx in range(first_idx, first_idx + seq_len_inner):

                    filename = filenames[file_idx]
                    # Images
                    _image = os.path.join(image_dir, filename)
                    assert os.path.isfile(_image)
                    seq_images.append(_image)

                    # Semantic Segmentation
                    if do_seg:
                        _mask = os.path.join(mask_dir, filename[:-4] + '.png')
                        assert os.path.isfile(_mask)
                        seq_masks.append(_mask)

                    # Deblurring
                    if do_deblur:
                        _deblur_frame = os.path.join(deblur_dir, filename[:-4] + '.png')
                        assert os.path.isfile(_deblur_frame)
                        seq_deblur_frames.append(_deblur_frame)

                    # Flows
                    if do_flow:

                        if file_idx==0:
                            _optical_flow = os.path.join(of_dir, filenames[file_idx+1][:-4] + '.npy')
                            assert os.path.isfile(_optical_flow)
                            seq_flows.append(_optical_flow)

                        else:
                            _optical_flow = os.path.join(of_dir, filename[:-4] + '.npy')
                            assert os.path.isfile(_optical_flow)
                            seq_flows.append(_optical_flow)



                self.images.append(seq_images)
                self.masks.append(seq_masks)
                self.deblur_frames.append(seq_deblur_frames)
                self.flows.append(seq_flows)

        # if seq_len is None:
        #     return
        # Uncomment to overfit to one sequence


        # Display stats
        print('Number of {} dataset sequences: {:d}'.format(self.split, len(self.images)))


    def __getitem__(self, index):

        _img = self._load_img(index)
        sample = {'image': _img}

        if self.do_seg:
            sample['segment'] = self._load_mask(index)

        if self.do_deblur:
            sample['deblur'] = self._load_deblur(index)

        if self.do_homo:
            sample['flow'] = self._load_flow(index)

        if self.meta:
            sample['meta'] = {'paths': self.images[index],
                              'im_size': (_img[0].shape[0], _img[0].shape[1]),
                              'transformations': {}}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _load_img(self, index):
        return [cv2.imread(path).astype(np.float32) for path in self.images[index]]

    def _load_deblur(self, index):
        return [cv2.imread(path).astype(np.float32) for path in self.deblur_frames[index]]

    def _load_mask(self, index):
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) for path in self.masks[index]]

    def _load_homo(self, index):
        return [np.load(path).astype(np.float32) for path in self.flows[index]]

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return 'DST Multitask (split=' + str(self.split) + ')'


