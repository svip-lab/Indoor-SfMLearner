# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))


class ScannetTestPoseDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 is_train=False,
                 ):
        super(ScannetTestPoseDataset, self).__init__()
        self.full_res_shape = (1296, 968) 
        self.K = self._get_intrinsics()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.resize = transforms.Resize(
                (self.height, self.width),
                interpolation=self.interp
        )

        self.load_depth = False

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        line = self.filenames[index].split()
        line = [os.path.join(self.data_path, item) for item in line]

        for ind, i in enumerate(self.frame_idxs):
            inputs[("color", i, -1)] = self.get_color(line[ind])

        K = self.K.copy()
        this_width = self.width 
        this_height= self.height 

        K[0, :] *= this_width 
        K[1, :] *= this_height

        inv_K = np.linalg.pinv(K)

        inputs[("K")] = torch.from_numpy(K).float()
        inputs[("inv_K")] = torch.from_numpy(inv_K).float()
        
        # self.preprocess(inputs)

        for i in self.frame_idxs:
            inputs[('color', i, 0)] = self.to_tensor(
                    self.resize(inputs[('color', i,  -1)])
            )
            del inputs[("color", i, -1)]
   

        if self.load_depth:
            for ind, i in enumerate(self.frame_idxs):
                this_depth = line[ind].replace('color', 'depth').replace('.jpg', '.png')
                this_depth = cv2.imread(this_depth, -1) / 1000.0
                this_depth = cv2.resize(this_depth, (self.width, self.height))
                this_depth = self.to_tensor(this_depth)

                # assume no flippling
                inputs[("depth", i)] = this_depth
        
        pose1_dir = line[0].replace('color', 'pose').replace('.jpg', '.txt')
        pose2_dir = line[1].replace('color', 'pose').replace('.jpg', '.txt')
        pose1 = np.loadtxt(pose1_dir, delimiter=' ')
        pose2 = np.loadtxt(pose2_dir, delimiter=' ')
        pose_gt = np.dot(np.linalg.inv(pose2), pose1)
        inputs['pose_gt'] = pose_gt
        
        return inputs

    def get_color(self, fp):
        color = self.loader(fp)
        return Image.fromarray(color)

    def check_depth(self):
        return False

    def _get_intrinsics(self):
        w, h = self.full_res_shape
        intrinsics =np.array([[1161.75/w, 0., 658.042/w, 0.], 
                               [0., 1169.11/h, 486.467/h, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")
        return intrinsics



class ScannetTestDepthDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
        ):
        super(ScannetTestDepthDataset, self).__init__()
        
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.interp = Image.ANTIALIAS

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.resize = transforms.Resize(
            (self.height, self.width),
            interpolation=self.interp

        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rgb = self.filenames[index].replace('/','_')
        rgb = os.path.join(self.data_path, rgb)
        depth = rgb.replace('color', 'depth').replace('jpg','png')
        
        rgb = self.loader(rgb)
        depth = cv2.imread(depth, -1) / 1000
       
        rgb = Image.fromarray(rgb)

        rgb = self.to_tensor(self.resize(rgb))
        depth = self.to_tensor(depth)
        
        return rgb, depth
