# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import random
import numpy as np
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms

import time
import h5py
CROP = 16

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic, watershed, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from datasets.extract_svo_point import PixelSelector

import ctypes
import multiprocessing as mp


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = np.array(img.convert('RGB'))
            h, w, c = img.shape
            return img

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    norm = np.array(h5f['norm'])
    norm = np.transpose(norm, (1,2,0))
    valid_mask = np.array(h5f['mask'])

    return rgb, depth, norm, valid_mask


class NYUDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 segment_path = '',
                 return_segment=False,
                 shared_dict=None):
        super(NYUDataset, self).__init__()
       
        self.debug = False
        self.return_segment = return_segment
        # remove 16 pixels from borders
        self.full_res_shape = (640-CROP*2, 480-CROP*2) 
        self.K = self._get_intrinsics()
        # print('The Normalized Intrinsics is ', self.K) 

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.segment_path = segment_path
        self.img_cache = shared_dict

        if self.is_train: self.loader = pil_loader
        else: self.loader = h5_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        self.pixelselector = PixelSelector()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                # import pdb; pdb.set_trace()
                for i in range(self.num_scales):                   
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        if not(self.is_train):
            line = self.filenames[index]
            line = os.path.join(self.data_path, line)
            rgb, depth, norm, valid_mask = self.loader(line) 
            h, w, c = rgb.shape

            rgb = rgb[44: 471, 40: 601, :]
            depth = depth[44: 471, 40: 601]
            # norm = norm[44:471, 40:601, :]
            # valid_mask = valid_mask[44:471, 40:601]

            rgb = Image.fromarray(rgb)
            depth = Image.fromarray(depth)

            rgb = self.to_tensor(self.resize[0](rgb))
            depth = self.to_tensor(self.resize[0](depth))
            # norm = self.to_tensor(norm)
            # norm_mask = self.to_tensor(valid_mask)

            K = self.K.copy()
            K[0, :] *= self.width
            K[1, :] *= self.height
            return rgb, depth, K, np.linalg.pinv(K)

        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        # line = [l.replace("/group/nyu_depth_v2", self.data_path) for l in line]
        line = [os.path.join(self.data_path, l) for l in line]
        for ind, i in enumerate(self.frame_idxs):
            if not i in set([0, -2, -1, 1, 2]):
                continue

            inputs[("color", i, -1)] = self.get_color(line[ind], do_flip)
            if self.debug:
                inputs[("color", i, -1)] = self.to_tensor(self.get_color(line[ind], do_flip))

        # load segments
        if self.return_segment:
            sp = self.segment_path + '/seg_%d.npz'%(index)
            if sp in self.img_cache:
                segments = self.img_cache[sp]
            else:
                segments = np.load(sp)
                self.img_cache[sp] = {'segment_0': segments['segment_0']}

            for ind, i in enumerate(self.frame_idxs):
                if not i in set([0]):
                    continue
    
                segment = segments['segment_%d'%(ind)]
                if do_flip:
                    segment = cv2.flip(segment, 1)
                inputs[('segment', i, 0)] = self.to_tensor(segment).long() + 1


        if self.debug:
            return inputs

        # inputs[("color", 0, -1)] = self.get_color(line[0], do_flip)

        # svo_map = np.zeros((480, 640)) 
        svo_map_resized = np.zeros((self.height, self.width)) # 288 * 384
        img = np.array(inputs[("color", 0, -1)])
        key_points = self.pixelselector.extract_points(img)                                                    
        key_points = key_points.astype(int)
        key_points[:,0] = key_points[:,0] * self.height // 480
        key_points[:,1] = key_points[:,1] * self.width // 640

        # noise 1000 points
        noise_num = 3000 - key_points.shape[0]
        noise_points = np.zeros((noise_num, 2), dtype=np.int32)
        noise_points[:, 0] = np.random.randint(self.height, size=noise_num)
        noise_points[:, 1] = np.random.randint(self.width, size=noise_num)

        svo_map_resized[key_points[:,0], key_points[:,1]] = 1

        inputs['svo_map'] = torch.from_numpy(svo_map_resized.copy())

        svo_map_resized[noise_points[:,0], noise_points[:,1]] = 1
        inputs['svo_map_noise'] = torch.from_numpy(
            svo_map_resized,
        ).float()

        keypoints = np.concatenate((key_points, noise_points), axis=0)
        inputs['dso_points'] = torch.from_numpy(
            keypoints,
        ).float()

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            if not i in set([0, -2, -1, 1, 2]):
                continue

            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
    
        return inputs

    def get_color(self, fp, do_flip):
        if fp in self.img_cache:
            color = self.img_cache[fp]
        else:
            color = self.loader(fp)
            if not(self.debug):
                color = self._undistort(color)
            self.img_cache[fp] = color

        if do_flip:
            color = cv2.flip(color, 1)
            # color = color.transpose(pil.FLIP_LEFT_RIGHT)
        # print(color.shape)
        h, w, c = color.shape
        color = color[CROP:h-CROP, CROP:w-CROP, :]

        return Image.fromarray(color)

    def check_depth(self):
        return False
        # raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def _get_intrinsics(self):
        # 640, 480
        w, h = self.full_res_shape
        
        fx = 5.1885790117450188e+02 / w
        fy = 5.1946961112127485e+02 / h
        cx = 3.2558244941119034e+02 / w
        cy = 2.5373616633400465e+02 / h

        intrinsics =np.array([[fx, 0., cx, 0.], 
                               [0., fy, cy, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")
        return intrinsics

    def _undistort(self, image):
        k1 =  2.0796615318809061e-01
        k2 = -5.8613825163911781e-01
        p1 = 7.2231363135888329e-04
        p2 = 1.0479627195765181e-03
        k3 = 4.9856986684705107e-01

        fx = 5.1885790117450188e+02
        fy = 5.1946961112127485e+02
        cx = 3.2558244941119034e+02
        cy = 2.5373616633400465e+02

        kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([[k1, k2, p1, p2, k3]])
        image = cv2.undistort(image, kmat, dist)
        return image


class NYUTestDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
        ):
        super(NYUTestDataset, self).__init__()
        self.full_res_shape = (640-CROP*2, 480-CROP*2) 
        self.K = self._get_intrinsics()
        
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.interp = Image.ANTIALIAS

        self.loader = h5_loader
        self.to_tensor = transforms.ToTensor()

        self.resize = transforms.Resize(
            (self.height, self.width),
            interpolation=self.interp
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        line = self.filenames[index]
        line = os.path.join(self.data_path, line)
        rgb, depth, norm, valid_mask = self.loader(line) 

        rgb = rgb[44: 471, 40: 601, :]
        depth = depth[44: 471, 40: 601]
        norm = norm[44:471, 40:601, :]
        valid_mask = valid_mask[44:471, 40:601]

        rgb = Image.fromarray(rgb)
        rgb = self.to_tensor(self.resize(rgb))

        depth = self.to_tensor(depth)
        norm = self.to_tensor(norm)
        norm_mask = self.to_tensor(valid_mask)

        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height
        return rgb, depth, norm, norm_mask, K, np.linalg.pinv(K)

    def _get_intrinsics(self):
        # 640, 480
        w, h = self.full_res_shape
        
        fx = 5.1885790117450188e+02 / w
        fy = 5.1946961112127485e+02 / h
        cx = 3.2558244941119034e+02 / w
        cy = 2.5373616633400465e+02 / h

        intrinsics =np.array([[fx, 0., cx, 0.], 
                               [0., fy, cy, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")
        return intrinsics


