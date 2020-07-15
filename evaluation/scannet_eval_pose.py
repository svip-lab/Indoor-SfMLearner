# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os,sys
sys.path.append(os.getcwd())
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from datasets import ScannetTestPoseDataset
import networks

from tqdm import tqdm

def compute_pose_errors(gt, pr):
    """from https://github.com/princeton-vl/DeepV2D/blob/master/evaluation/eval_utils.py
    """
    # seperate rotations and translations
    R1, t1 = gt[:3, :3], gt[:3, 3]
    R2, t2 = pr[:3, :3], pr[:3, 3]

    costheta = (np.trace(np.dot(R1.T, R2))-1.0)/2.0
    costheta = np.minimum(costheta, 1.0)
    rdeg = np.arccos(costheta) * (180/np.pi)

    t1mag = np.sqrt(np.dot(t1,t1))
    t2mag = np.sqrt(np.dot(t2,t2))
    costheta = np.dot(t1,t2) / (t1mag*t2mag)
    tdeg = np.arccos(costheta) * (180/np.pi)

    # fit scales to translations
    a = np.dot(t1, t2) / np.dot(t2, t2)
    tcm = 100*np.sqrt(np.sum((t1-a*t2)**2, axis=-1))

    if np.isnan(rdeg) or np.isnan(tdeg) or np.isnan(tcm):
        raise ValueError
    return rdeg, tdeg, tcm

def prepare_model_for_test(opt):
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))
    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)

    pose_encoder.load_state_dict(torch.load(pose_encoder_path))
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))
    
    pose_encoder.cuda().eval()
    pose_decoder.cuda().eval()

    return pose_encoder, pose_decoder


def evaluate(opt):
    pose_errors = []
    pose_encoder, pose_decoder = prepare_model_for_test(opt)

    filenames = readlines('./splits/scannet_test_pose_deepv2d.txt')
    dataset = ScannetTestPoseDataset(
            opt.data_path, 
            filenames, 
            opt.height, opt.width,
            frame_idxs = opt.frame_ids,
    )

    dataloader = DataLoader(
            dataset, 1, shuffle=False,
            num_workers=opt.num_workers, 
    )

    print("-> Computing pose predictions")

    with torch.no_grad():
        for ind, inputs in enumerate(tqdm(dataloader)):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            color = torch.cat(
                    [inputs[("color", i, 0)] for i in opt.frame_ids], 
                    axis = 1,
            )
            features = pose_encoder(color)
            axisangle, translation = pose_decoder([features])
            this_pose = transformation_from_parameters( 
                    axisangle[:, 0], 
                    translation[:, 0]
            )
            this_pose = this_pose.data.cpu().numpy()[0]
            gt_pose = inputs['pose_gt'].data.cpu().numpy()[0]
            pose_errors.append(compute_pose_errors(this_pose, gt_pose))
    
    mean_pose_errors = np.array(pose_errors).mean(0)
    print("\n  " + ("{:>8} | " * 3).format("rot", "tdeg", "tcm"))
    print(("&{: 8.3f}  " * 3).format(*mean_pose_errors.tolist()) + "\\\\")
    print("\n-> Done!")



if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
