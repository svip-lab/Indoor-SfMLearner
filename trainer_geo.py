# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import datasets
import numpy as np
import time
import weakref

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import json

from utils import *
from layers import *

import datasets
import networks
import random

from multiprocessing import Manager
# Init, get rid of slow io
manager = Manager()
shared_dict = manager.dict()

# seed
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

# from IPython import embed

# copy from https://github.com/vcg-uvic/linearized_multisampling_release/blob/master/warp/linearized.py
######### Utils to minimize dependencies #########
# Move utils to another file if you want
def print_notification(content_list, notification_type='NOTIFICATION'):
    print('---------------------- {0} ----------------------'.format(notification_type))
    print()
    for content in content_list:
        print(content)
    print()
    print('----------------------------------------------------')


def is_nan(x):
    '''
    get mask of nan values.
    :param x: torch or numpy var.
    :return: a N-D array of bool. True -> nan, False -> ok.
    '''
    return x != x


def has_nan(x) -> bool:
    '''
    check whether x contains nan.
    :param x: torch or numpy var.
    :return: single bool, True -> x containing nan, False -> ok.
    '''
    return is_nan(x).any()


def mat_3x3_inv(mat):
    '''
    calculate the inverse of a 3x3 matrix, support batch.
    :param mat: torch.Tensor -- [input matrix, shape: (B, 3, 3)]
    :return: mat_inv: torch.Tensor -- [inversed matrix shape: (B, 3, 3)]
    '''
    if len(mat.shape) < 3:
        mat = mat[None]
    assert mat.shape[1:] == (3, 3)

    # Divide the matrix with it's maximum element
    max_vals = mat.max(1)[0].max(1)[0].view((-1, 1, 1))
    mat = mat / max_vals

    det = mat_3x3_det(mat)
    inv_det = 1.0 / det

    mat_inv = torch.zeros(mat.shape, device=mat.device)
    mat_inv[:, 0, 0] = (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 2, 1] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 0, 1] = (mat[:, 0, 2] * mat[:, 2, 1] - mat[:, 0, 1] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 0, 2] = (mat[:, 0, 1] * mat[:, 1, 2] - mat[:, 0, 2] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 1, 0] = (mat[:, 1, 2] * mat[:, 2, 0] - mat[:, 1, 0] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 1, 1] = (mat[:, 0, 0] * mat[:, 2, 2] - mat[:, 0, 2] * mat[:, 2, 0]) * inv_det
    mat_inv[:, 1, 2] = (mat[:, 1, 0] * mat[:, 0, 2] - mat[:, 0, 0] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 2, 0] = (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 2, 0] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 2, 1] = (mat[:, 2, 0] * mat[:, 0, 1] - mat[:, 0, 0] * mat[:, 2, 1]) * inv_det
    mat_inv[:, 2, 2] = (mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 1, 0] * mat[:, 0, 1]) * inv_det

    # Divide the maximum value once more
    mat_inv = mat_inv / max_vals
    return mat_inv 


def mat_3x3_det(mat):
    '''
    calculate the determinant of a 3x3 matrix, support batch.
    '''
    if len(mat.shape) < 3:
        mat = mat[None]
    assert mat.shape[1:] == (3, 3)

    det = mat[:, 0, 0] * (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 2, 1] * mat[:, 1, 2]) \
        - mat[:, 0, 1] * (mat[:, 1, 0] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 0]) \
        + mat[:, 0, 2] * (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 1, 1] * mat[:, 2, 0])
    return det


def inv_SE3(G):
    """Inverts rigid body transformation"""
    batch, _, _ = G.size()                                                                                  
    R = torch.transpose(G[:, 0:3, 0:3], 1, 2).contiguous()
    t = G[:, 0:3, 3].view(batch, 3, 1)
    tp = -torch.matmul(R, t)

    filler = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4).astype(np.float32)
    filler = torch.Tensor(filler).repeat(batch, 1, 1).to(G.device)

    Ginv = torch.cat([torch.cat([R, tp], dim=2).float(), filler], dim=1)
    return Ginv

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


class Trainer:
    def __init__(self, options):

        self.opt = options

        self.debug = self.opt.debug
        print('DEBUG: ', self.debug)

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = True

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
 
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        print("Training is using frames: \n  ", self.opt.frame_ids_to_train)

        # data
        datasets_dict = {"nyu": datasets.NYUDataset }
        self.dataset = datasets_dict[self.opt.dataset]

        train_filenames = readlines('./splits/nyu_train_0_10_20_30_40.txt')

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 1, is_train=True, 
            segment_path=self.opt.segment_path,
            return_segment=True,
            shared_dict=shared_dict)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        # validation
        filenames = readlines('./splits/nyu_test.txt')
        # filenames = [filename.replace("/p300/Code/self_depth/monodepth2/nyuv2/nyu_official",
        #                               self.opt.val_path) for filename in filenames]
        val_dataset = datasets.NYUDataset(self.opt.val_path, filenames,
                                          self.opt.height, self.opt.width,
                                          [0], 1, is_train=False, return_segment=False)
        self.val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=2)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim_sparse = SSIM_sparse()
        self.ssim_sparse.to(self.device)

        self.backproject_depth = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), -1))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.val()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            self.val()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()
        for param in self.model_optimizer.param_groups:
            print("lr:", param["lr"])

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 400 == 0

            # if self.step % 5 == 0:
            self.log_time(batch_idx, duration, losses)

            if early_phase or late_phase:
                self.log("train", inputs, outputs, losses)

            for items in outputs.items():
                del items

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
           Inputs -> dict consists of :
            K/inv_K at 0~3 at 4 different scales
            color 0,  0~3 at 4 scales
            color 1,  0~3
            color -1, 0~3
        
            and color augmented versions   
        """
            
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}
        for i in [0]:
            features = self.models["encoder"](inputs[("color_aug", i, 0)])
            output = self.models["depth"](features)
            output = {(disp, i, scale): output[(disp, scale)] for (disp, scale) in output.keys()}
            outputs.update(output)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_sparse_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                # pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids_to_train}

            assert self.opt.frame_ids == [0, -4, -3, -2, -1, 1, 2, 3, 4]
            for f_i in [-2, -1, 0, 1] if len(self.opt.frame_ids_to_train) == 5 else [-1, 0]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    pose_inputs = [pose_feats[f_i], pose_feats[f_i + 1]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", f_i, f_i + 1)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)

            if len(self.opt.frame_ids_to_train) == 5: 
                outputs[("cam_T_cam", 0, 2)] = outputs[("cam_T_cam", 0, 1)] @ outputs[("cam_T_cam", 1, 2)]
                outputs[("cam_T_cam", -2, 0)] = outputs[("cam_T_cam", -2, -1)] @ \
                                                outputs[("cam_T_cam", -1, 0)]
                outputs[("cam_T_cam", 0, -2)] = inv_SE3(outputs[("cam_T_cam", -2, 0)])

            outputs[("cam_T_cam", 0, -1)] = inv_SE3(outputs[("cam_T_cam", -1, 0)])

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        errors = []
        with torch.no_grad():
            for ind, (data, gt_depth, K, K_inv) in enumerate(tqdm(self.val_dataloader)):
                input_color = data.cuda()

                output = self.models["depth"](self.models["encoder"](input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                pred_depth = 1 / pred_disp

                pred_depth = pred_depth[0]

                gt_depth = gt_depth.data.numpy()[0, 0]

                mask = gt_depth > 0
                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]

                ratio = np.median(gt_depth) / np.median(pred_depth)
                pred_depth *= ratio

                pred_depth[pred_depth < self.opt.min_depth] = self.opt.min_depth
                pred_depth[pred_depth > self.opt.max_depth] = self.opt.max_depth

                errors.append(compute_errors(gt_depth, pred_depth))

        mean_errors = np.array(errors).mean(0)
 
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")

        # write to tensorboard
        writer = self.writers["val"]
        for l, v in zip(["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"],
                        mean_errors.tolist()):
            writer.add_scalar("{}".format(l), v, self.step)
        # writer.flush()
        self.set_train()

    def generate_planar_depth(self, inputs, outputs, frame_id, scale):                       
        source_scale = 0
        depth = outputs[("depth", frame_id, scale)]

        cam_points = self.backproject_depth[source_scale](                                   
            depth, inputs[("inv_K", source_scale)])

        # superpixel pooling, superpixel index start from one
        # todo currently only use one scale segment results
        superpixel = inputs[('segment', frame_id, 0)].long() - 1
        max_num = superpixel.max().item() + 1

        sum_points = torch.zeros((self.opt.batch_size, max_num, 3)).to(self.device)
        area = torch.zeros((self.opt.batch_size, max_num)).to(self.device)
        for channel in range(3):
            points_channel = sum_points[:, :, channel]
            points_channel = points_channel.reshape(self.opt.batch_size, -1)
            points_channel.scatter_add_(1, superpixel.view(self.opt.batch_size, -1),
                                        cam_points[:, channel, ...].view(self.opt.batch_size, -1))

        area.scatter_add_(1, superpixel.view(self.opt.batch_size, -1),
                          torch.ones_like(depth).view(self.opt.batch_size, -1))

        # X^T X
        cam_points_tmp = cam_points[:, :3, :]
        x_T_dot_x = (cam_points_tmp.unsqueeze(1) * cam_points_tmp.unsqueeze(2)).view(self.opt.batch_size, 9, -1)
        X_T_dot_X = torch.zeros((self.opt.batch_size, max_num, 9)).cuda()
        for channel in range(9):
            points_channel = X_T_dot_X[:, :, channel]
            points_channel = points_channel.reshape(self.opt.batch_size, -1)
            points_channel.scatter_add_(1, superpixel.view(self.opt.batch_size, -1),
                                        x_T_dot_x[:, channel, ...].view(self.opt.batch_size, -1))
        xTx = X_T_dot_X.view(self.opt.batch_size, max_num, 3, 3)

        # take inverse
        xTx_inv = mat_3x3_inv(xTx.view(-1, 3, 3) + 0.01*torch.eye(3).view(1,3,3).expand(self.opt.batch_size*max_num, 3, 3).cuda())
        xTx_inv = xTx_inv.view(xTx.shape)

        xTx_inv_xT = torch.matmul(xTx_inv, sum_points.unsqueeze(3))
        plane_parameters = xTx_inv_xT.squeeze(3)
        
        # outputs[("fitted_parameters", frame_id, scale)] = plane_parameters

        # generate mask for superpixel with area larger than 200
        valid_mask = ( area > 1000. ).float()
        planar_mask = torch.gather(valid_mask, 1, superpixel.view(self.opt.batch_size, -1))
        planar_mask = planar_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)
        outputs[("planar_mask", frame_id, scale)] = planar_mask
        
        # superpixel unpooling
        unpooled_parameters = []
        for channel in range(3):
            pooled_parameters_channel = plane_parameters[:, :, channel]
            pooled_parameters_channel = pooled_parameters_channel.reshape(self.opt.batch_size, -1)
            unpooled_parameter = torch.gather(pooled_parameters_channel, 1, superpixel.view(self.opt.batch_size, -1))
            unpooled_parameters.append(unpooled_parameter.view(self.opt.batch_size, 1, self.opt.height, self.opt.width))
        unpooled_parameters = torch.cat(unpooled_parameters, dim=1)


        # recover depth from plane parameters
        K_inv_dot_xy1 = torch.matmul(inputs[("inv_K", source_scale)][:, :3, :3],
                                     self.backproject_depth[source_scale].pix_coords)
        depth = 1. / (torch.sum(K_inv_dot_xy1 * unpooled_parameters.view(self.opt.batch_size, 3, -1), dim=1) + 1e-6)

        # clip depth range
        depth = torch.clamp(depth, self.opt.min_depth, self.opt.max_depth)
        depth = depth.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)
        outputs[("planar_depth", frame_id, scale)] = depth

    def generate_sparse_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
     
        for scale in self.opt.scales:
            disp = outputs[("disp", 0, scale)]
            disp = F.interpolate(
                      disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            self.generate_planar_depth(inputs, outputs, 0, scale)

            # sample depth for dso points                                                
            dso_points = inputs['dso_points']
            y0 = dso_points[:, :, 0]
            x0 = dso_points[:, :, 1]
            dso_points = torch.cat((x0.unsqueeze(2), y0.unsqueeze(2)), dim=2)

            flat = (x0 + y0 * self.opt.width).long()
            dso_depth = torch.gather(depth.view(self.opt.batch_size, -1), 1, flat)

            # generate pattern
            meshgrid = np.meshgrid([-2, 0, 2],[-2, 0, 2], indexing='xy')
            meshgrid = np.stack(meshgrid, axis=0).astype(np.float32)
            meshgrid = torch.from_numpy(meshgrid).to(dso_points.device).permute(1, 2, 0).view(1, 1, 9, 2)
            dso_points = dso_points.unsqueeze(2) + meshgrid
            dso_points = dso_points.reshape(self.opt.batch_size, -1, 2)
            dso_depth = dso_depth.view(self.opt.batch_size, -1, 1).expand(-1, -1, 9).reshape(self.opt.batch_size, 1, -1)

            # convert to point cloud
            xy1 = torch.cat((dso_points, torch.ones_like(dso_points[:, :, :1])), dim=2)
            xy1 = xy1.permute(0, 2, 1)
            cam_points = (inputs[("inv_K", source_scale)][:, :3, :3] @ xy1) * dso_depth
            points = torch.cat((cam_points, torch.ones_like(cam_points[:, :1, :])), dim=1)
            outputs[("cam_T_cam", 0, 0)] = torch.eye(4).view(1, 4, 4).expand(self.opt.batch_size, 4, 4).cuda()

            for _, frame_id in enumerate(self.opt.frame_ids_to_train):
                T = outputs[("cam_T_cam", 0, frame_id)]

                # projects to different frames
                P = torch.matmul(inputs[("K", source_scale)], T)[:, :3, :]
                cam_points = torch.matmul(P, points)

                pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
                pix_coords = pix_coords.view(self.opt.batch_size, 2, -1, 9)
                pix_coords = pix_coords.permute(0, 2, 3, 1)
                pix_coords[..., 0] /= self.opt.width - 1
                pix_coords[..., 1] /= self.opt.height - 1
                pix_coords = (pix_coords - 0.5) * 2

                # save mask
                valid = (pix_coords[..., 0] > -1.) & (pix_coords[..., 0] < 1.) & (pix_coords[..., 1] > -1.) & (
                            pix_coords[..., 1] < 1.)
                outputs[("dso_mask", frame_id, scale)] = valid.unsqueeze(1).float()

                # sample patch from color images
                outputs[("dso_color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border")

    def patch_sampler(self, color, pix_coords, padding_mode="border"):
        meshgrid = np.meshgrid([-1, 0, 1],[-1, 0, 1], indexing='xy')
        meshgrid = np.stack(meshgrid, axis=0).astype(np.float32)
        meshgrid = torch.from_numpy(meshgrid).to(pix_coords.device).permute(1, 2, 0).view(-1, 2)
        meshgrid[:, 0] /= self.opt.width
        meshgrid[:, 1] /= self.opt.height
        meshgrid *= 2
        meshgrid = meshgrid.view(1, 1, 9, 2)

        pix_coords = pix_coords + meshgrid

        output = F.grid_sample(                         
                    color,
                    pix_coords,
                    padding_mode=padding_mode)
        return output

    def compute_sparse_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        # todo use middle values?
        l1_loss = l1_loss.mean(3, True)
        ssim_loss = self.ssim_sparse(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            sparse_reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            dso_target = outputs[("dso_color", 0, scale)]

            assert self.opt.frame_ids == [0, -4, -3, -2, -1, 1, 2, 3, 4]

            # dso loss
            assert self.opt.frame_ids_to_train[0] == 0
            for frame_id in self.opt.frame_ids_to_train[1:]:
                dso_pred = outputs[("dso_color", frame_id, scale)]
                sparse_reprojection_losses.append(self.compute_sparse_reprojection_loss(dso_pred, dso_target))

            if len(self.opt.frame_ids_to_train) == 5:
                dso_combined_1 = torch.cat((sparse_reprojection_losses[1], sparse_reprojection_losses[2]), dim=1)
                dso_combined_2 = torch.cat((sparse_reprojection_losses[0], sparse_reprojection_losses[3]), dim=1)

                dso_to_optimise_1, _ = torch.min(dso_combined_1, dim=1)
                dso_to_optimise_2, _ = torch.min(dso_combined_2, dim=1)
                dso_loss_1 = dso_to_optimise_1.mean() 
                dso_loss_2 = dso_to_optimise_2.mean()

                loss += dso_loss_1 + dso_loss_2

                losses["dso_loss_1/{}".format(scale)] = dso_loss_1
                losses["dso_loss_2/{}".format(scale)] = dso_loss_2
            else:
                dso_combined_1 = torch.cat(sparse_reprojection_losses, dim=1)
                dso_to_optimise_1, _ = torch.min(dso_combined_1, dim=1)
                dso_loss_1 = dso_to_optimise_1.mean()
                loss += dso_loss_1 

                losses["dso_loss_1/{}".format(scale)] = dso_loss_1

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            losses["smooth_loss/{}".format(scale)] = smooth_loss

            # planar depth
            loss_planar_reg = 0.0
            for frame_id in [0]:
                pred_depth = outputs[("depth", frame_id, scale)]
                planar_depth = outputs[("planar_depth", frame_id, scale)]
                planar_mask = outputs[("planar_mask", frame_id, scale)] 

                loss_planar_reg += torch.mean(torch.abs(pred_depth - planar_depth) * planar_mask)
            loss += loss_planar_reg * self.opt.lambda_planar_reg
            losses["planar_reg_loss/{}".format(scale)] = loss_planar_reg

            total_loss += loss

            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss"].cpu().data,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

        writer = self.writers["train"]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        
        # import pdb; pdb.set_trace()
        
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            writer.add_image(
                "svo_{}/{}".format(0, j), 
                 inputs['svo_map'][j].unsqueeze(0).data, self.step)
            writer.add_image(
                "svo_noise_{}/{}".format(0, j), 
                 inputs['svo_map_noise'][j].unsqueeze(0).data, self.step)

            # for s in self.opt.scales:
            for s in [0]:
                for frame_id in [0, -1, 1]:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if frame_id == 0:
                        writer.add_image(
                            "disp_{}_{}/{}".format(frame_id, s, j),
                            normalize_image(outputs[("disp", frame_id, s)][j]), self.step)
                        writer.add_image(
                            "depth_{}_{}/{}".format(frame_id, s, j),
                            normalize_image(outputs[("depth", frame_id, s)][j]), self.step)

                '''
                writer.add_image(                                                            
                    "planar_depth_{}/{}".format(s, j),
                    normalize_image(torch.clamp(outputs[("planar_depth", 0, s)][j], outputs[("depth", frame_id, s)][j].min().item(), outputs[("depth", frame_id, s)][j].max().item())), self.step)
                
                writer.add_image(
                    "planar_mask_{}/{}".format(s, j),
                    outputs[("planar_mask", 0, s)][j], self.step)
                '''

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

