import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import random


class ColorJitter(object):
    def __init__(self):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = [0.2, 0.15, 0.3, 0.1]

    def __call__(self, sample):

        if 'image' in sample.keys() and 'deblur' in sample.keys():

            sample.update({'image': [Image.fromarray(np.uint8(_image)) for _image in sample['image']]})
            sample.update({'deblur': [Image.fromarray(np.uint8(_deblur)) for _deblur in sample['deblur']]})

            if self.brightness > 0:
                brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                sample.update({'image': [F.adjust_brightness(_image, brightness_factor) for _image in sample['image']]})
                sample.update({'deblur': [F.adjust_brightness(_deblur, brightness_factor) for _deblur in sample['deblur']]})
                sample['meta']['transformations']['brightness'] = brightness_factor

            if self.contrast > 0:
                contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                sample.update({'image': [F.adjust_contrast(_image, contrast_factor) for _image in sample['image']]})
                sample.update({'deblur': [F.adjust_contrast(_deblur, contrast_factor) for _deblur in sample['deblur']]})
                sample['meta']['transformations']['contrast'] = contrast_factor

            if self.saturation > 0:
                saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                sample.update({'image': [F.adjust_saturation(_image, saturation_factor) for _image in sample['image']]})
                sample.update({'deblur': [F.adjust_saturation(_deblur, saturation_factor) for _deblur in sample['deblur']]})
                sample['meta']['transformations']['saturation'] = saturation_factor

            if self.hue > 0:
                hue_factor = np.random.uniform(-self.hue, self.hue)
                sample.update({'image': [F.adjust_hue(_image, hue_factor) for _image in sample['image']]})
                sample.update({'deblur': [F.adjust_hue(_deblur, hue_factor) for _deblur in sample['deblur']]})
                sample['meta']['transformations']['hue'] = hue_factor

            sample.update({'image': [np.asarray(_image).clip(0, 255) for _image in sample['image']]})
            sample.update({'deblur': [np.asarray(_deblur).clip(0, 255) for _deblur in sample['deblur']]})

        return sample


class RandomColorChannel(object):
    def __call__(self, sample):
        random_order = np.random.permutation(3)
        # only apply to blurry and gt images
        if 'image' in sample.keys() and 'deblur' in sample.keys() and random.random() < 0.5:
            sample.update({'image': [_image[:, :, random_order] for _image in sample['image']]})
            sample.update({'deblur': [_deblur[:, :, random_order] for _deblur in sample['deblur']]})
            sample['meta']['transformations']['channels_order'] = ''.join(['bgr'[i] for i in random_order])
        else:
            sample['meta']['transformations']['channels_order'] = 'bgr'
        return sample


class RandomGaussianNoise(object):
    def __init__(self):
        gaussian_para = [0, 8]
        self.mu = gaussian_para[0]
        self.std_var = gaussian_para[1]

    def __call__(self, sample):

        shape = sample['image'][0].shape
        gaussian_noise = np.random.normal(self.mu, self.std_var, shape).astype(np.float32)
        # only apply to blurry images
        sample.update({'image': [(_image + gaussian_noise).clip(0, 255) for _image in sample['image']]})
        sample['meta']['transformations']['gaussian_noise'] = gaussian_noise
        return sample


class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, dims=(640, 480)):
        self.dims = dims

    @staticmethod
    def homography_resize(homography_offsets, orig_dims, new_dims):
        homography, offsets = homography_offsets
        w_ratio = new_dims[0]/orig_dims[0]
        h_ratio = new_dims[1]/orig_dims[1]
        new_offsets = offsets * np.asarray([w_ratio, h_ratio])
        f = np.eye(3)
        f[0, 0] = w_ratio
        f[1, 1] = h_ratio
        new_homo = np.matmul(np.matmul(f, homography), np.linalg.inv(f))
        return new_homo, new_offsets

    def __call__(self, sample):
        #  Homography has to be changed first because it needs original image size
        if 'homography' in sample.keys():
            sample['homography'] = [self.homography_resize(homo_offsets, (sample['image'][num].shape[1],
                                                                  sample['image'][num].shape[0]), self.dims)
                                    for num, homo_offsets in enumerate(sample['homography'])]

        # Apply to gt, blurry and masks.
        sample.update({'image': [cv2.resize(_image, self.dims, cv2.INTER_AREA) for _image in sample['image']]})

        if 'segment' in sample.keys():
            sample.update({'segment': [cv2.resize(_mask, self.dims, cv2.INTER_AREA) for _mask in sample['segment']]})
        if 'deblur' in sample.keys():  # TODO resizing affects the blur - ONLY FOR DEBUGGING
            sample.update({'deblur': [cv2.resize(_deblur, self.dims, cv2.INTER_AREA) for _deblur in sample['deblur']]})
        sample['meta']['transformations']['resize'] = self.dims
        return sample


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""

    @staticmethod
    def homography_horizontal_flip(homography_offsets, image_shape):
        homography, offsets = homography_offsets
        f = np.eye(3)
        f[0, 0] = -1
        new_homo = np.matmul(np.matmul(f, homography), f)
        new_homo[0, 2] = new_homo[0, 2] + (1 - new_homo[0, 0]) * image_shape[1]
        new_homo[1, 2] = new_homo[1, 2] + new_homo[0, 1] * image_shape[1]
        new_offsets = offsets.copy()
        new_offsets[0] = offsets[1]
        new_offsets[1] = offsets[0]
        new_offsets[2] = offsets[3]
        new_offsets[3] = offsets[2]
        new_offsets[:, 0] = image_shape[1] - new_offsets[:, 0]
        return new_homo, new_offsets

    def __call__(self, sample):
        # Apply to gt, blurry and masks. Additionally, change the homography
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''

            sample.update({'image': [np.copy(np.fliplr(_image)) for _image in sample['image']]})
            if 'deblur' in sample.keys():
                sample.update({'deblur': [np.copy(np.fliplr(_deblur)) for _deblur in sample['deblur']]})
            if 'segment' in sample.keys():
                sample.update({'segment': [np.copy(np.fliplr(_mask)) for _mask in sample['segment']]})
            if 'homography' in sample.keys():
                sample['homography'] = [self.homography_horizontal_flip(homo_offsets, sample['image'][num].shape) for num, homo_offsets
                                        in enumerate(sample['homography'])]
            sample['meta']['transformations']['horizontal_flip'] = True
        else:
            sample['meta']['transformations']['horizontal_flip'] = False
        return sample


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""

    @staticmethod
    def homography_vertical_flip(homography_offsets, image_shape):
        homography, offsets = homography_offsets
        f = np.eye(3)
        f[1, 1] = -1
        new_homo = np.matmul(np.matmul(f, homography), f)
        new_homo[0, 2] = new_homo[0, 2] - new_homo[0, 1] * image_shape[0]
        new_homo[1, 2] = new_homo[1, 2] + (1 - new_homo[0, 0]) * image_shape[0]
        new_offsets = offsets.copy()
        new_offsets[0] = offsets[2]
        new_offsets[1] = offsets[3]
        new_offsets[2] = offsets[0]
        new_offsets[3] = offsets[1]
        new_offsets[:, 1] = image_shape[0] - new_offsets[:, 1]
        return new_homo, new_offsets

    def __call__(self, sample):
        # Apply to gt, blurry and masks. Additionally, change the homography
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''

            sample.update({'image': [np.copy(np.flipud(_image)) for _image in sample['image']]})
            if 'deblur' in sample.keys():
                sample.update({'deblur': [np.copy(np.flipud(_deblur)) for _deblur in sample['deblur']]})
            if 'segment' in sample.keys():
                sample.update({'segment': [np.copy(np.flipud(_mask)) for _mask in sample['segment']]})
            if 'homography' in sample.keys():
                sample['homography'] = [self.homography_vertical_flip(homo_offsets, sample['image'][num].shape) for num, homo_offsets
                                        in enumerate(sample['homography'])]
            sample['meta']['transformations']['vertical_flip'] = True
        else:
            sample['meta']['transformations']['vertical_flip'] = False
        return sample


def homography_crop(homography_offsets, new_dims, x_start, y_start):
    homography, offsets = homography_offsets
    f = np.eye(3)
    f[0, 2] = -x_start
    f[1, 2] = -y_start
    new_homo = np.matmul(np.matmul(f, homography), np.linalg.inv(f))
    new_corners = np.asarray([[0, 0, 1], [new_dims[0], 0, 1], [0, new_dims[1], 1], [new_dims[0], new_dims[1], 1]])
    new_offsets = np.matmul(new_homo, new_corners.T)[:2].T
    return new_homo, new_offsets


class CenterCrop(object):
    """Resize image and/or masks."""

    def __init__(self, crop_size=(1280, 800)):
        self.crop_size_h = crop_size[1]
        self.crop_size_w = crop_size[0]

    def __call__(self, sample):

        input_size_h, input_size_w = sample['image'][0].shape[:2]
        x_start = int(round((input_size_w - self.crop_size_w) / 2.))
        y_start = int(round((input_size_h - self.crop_size_h) / 2.))

        #  Homography has to be changed first because it needs original image size
        if 'homography' in sample.keys():
            sample['homography'] = [homography_crop(homo_offsets, (sample['image'][num].shape[1], sample['image'][num].shape[0]),
                                                    (self.crop_size_w, self.crop_size_h), x_start, y_start)
                                    for num, homo_offsets in enumerate(sample['homography'])]

        sample['image'] = [_image[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for _image in sample['image']]

        if 'segment' in sample.keys():
            sample['segment'] = [_image[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for _image in sample['segment']]
        if 'deblur' in sample.keys():
            sample['deblur'] = [_image[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for _image in sample['deblur']]
        sample['meta']['transformations']['center_crop'] = (self.crop_size_h, self.crop_size_w)
        return sample


class RandomCrop(object):
    """Resize image and/or masks."""

    def __init__(self, crop_size=(256, 256), rho=16):
        self.crop_size_h = crop_size[1]
        self.crop_size_w = crop_size[0]
        self.rho = rho

    def __call__(self, sample):

        input_size_h, input_size_w = sample['image'][0].shape[:2]
        x_start = random.randint(self.rho, input_size_w - self.crop_size_w-1-self.rho)
        y_start = random.randint(self.rho, input_size_h - self.crop_size_h-1-self.rho)

        #  Homography has to be changed first because it needs original image size
        if 'homography' in sample.keys():
            sample['homography'] = [homography_crop(homo_offsets, (sample['image'][num].shape[1], sample['image'][num].shape[0]),
                                                    (self.crop_size_w, self.crop_size_h), x_start, y_start)
                                    for num, homo_offsets in enumerate(sample['homography'])]

        sample['image'] = [_image[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for _image in sample['image']]
        if 'segment' in sample.keys():
            sample['segment'] = [_image[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for _image in sample['segment']]
        if 'deblur' in sample.keys():
            sample['deblur'] = [_image[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for _image in sample['deblur']]
        sample['meta']['transformations']['random_crop'] = (self.crop_size_h, self.crop_size_w)
        return sample


class ToGrayscale(object):
    """Resize image and/or masks."""

    @staticmethod
    def to_grayscale(image):
        image = np.expand_dims(image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114, axis=-1)
        return image

    def __call__(self, sample):

        sample['image'] = [self.to_grayscale(_image) for _image in sample['image']]
        if 'deblur' in sample.keys():
            sample['deblur'] = [self.to_grayscale(_image) for _image in sample['deblur']]
        sample['meta']['transformations']['grayscale'] = True
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        sample.update({'image': [torch.from_numpy(_image).permute(2, 0, 1) for _image in sample['image']]})
        if 'segment' in sample.keys():
            sample.update({'segment': [torch.from_numpy((_mask > 125).astype(float)).type(torch.LongTensor) for _mask in sample['segment']]})
        if 'deblur' in sample.keys():
            sample.update({'deblur': [torch.from_numpy(_deblur).permute(2, 0, 1) for _deblur in sample['deblur']]})
        if 'homography' in sample.keys():
            sample.update({'homography': [[torch.from_numpy(_homo[0]).type(torch.FloatTensor),
                                           torch.from_numpy(_homo[1]).type(torch.FloatTensor)] for _homo in sample['homography']]})

        return sample


class Normalize(object):
    """Normalize image"""

    def __call__(self, sample):

        sample.update({'image': [_image/255 for _image in sample['image']]})
        if 'deblur' in sample.keys():
            sample.update({'deblur': [_deblur/255 for _deblur in sample['deblur']]})
        # Change from absolute positions to offsets
        if 'homography' in sample.keys():
            sample['homography'] = [[_homo[0], _homo[1] -
                                     torch.tensor([[0,0], [sample['image'][0].shape[2], 0],
                                                   [0, sample['image'][0].shape[1]],
                                                   [sample['image'][0].shape[2], sample['image'][0].shape[1]]])]
                                    for _homo in sample['homography']]
        return sample


class Standarize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        sample.update({'image': [(_image-self.mean) / self.std for _image in sample['image']]})

        if 'deblur' in sample.keys():
            sample.update({'deblur': [(_deblur-self.mean) / self.std for _deblur in sample['deblur']]})

        return sample
