# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default='')
        self.parser.add_argument("--segment_path",
                                 type=str,
                                 help="path to the segment",
                                 default='')
        self.parser.add_argument("--val_path",
                                 type=str,
                                 help="path to the testing set",
                                 default='')
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(file_dir, "logs"))

        self.parser.add_argument("--debug", default=False, help="debuginfo")

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 default="nyu")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="nyu",
                                 choices=["nyu"])
        self.parser.add_argument("--png", default=True,
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 )

        # 192, 640                         
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=384)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=10.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames order of training list file",
                                 default=[0, -4, -3, -2, -1, 1, 2, 3, 4])
        self.parser.add_argument("--frame_ids_to_train",                       
                                 nargs="+",
                                 type=int,
                                 help="frames to train",
                                 default=[0, -2, -1, 1, 2])
#                                 choices=[[0, -1, 1],
#                                          [0, -2, -1, 1, 2]])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=41)
        self.parser.add_argument("--scheduler_step_size",
                                 nargs="+",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=[26, 36])

        # ABLATION options
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=50)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

        # eval write depth output
        self.parser.add_argument("--eval_save_depth",
                                 help="if set will save depth to tensorboard",
                                 action="store_true")
        self.parser.add_argument("--eval_save_plyfile",
                                 help="if set will save depth to ply files",
                                 action="store_true")

        self.parser.add_argument("--lambda_planar_reg",                                      
                                 help="weights for planar consistency when train depth",
                                 type=float,
                                 default=0.05)


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
