from __future__ import absolute_import, division, print_function

import os, sys
sys.path.append(os.getcwd())
# import cv2
# cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
import numpy as np

import torch
import torch.nn.functional as F
import datasets
import networks

from tqdm import tqdm
from torch.utils.data import DataLoader

from layers import disp_to_depth, BackprojectDepth
from utils import readlines
from options import MonodepthOptions


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    _, l = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w)) 
    # l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - torch.clamp(20 * (l - 0.05), 0, 1)) # [None, ...]
    r_mask = torch.flip(l_mask, [1])
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def depth_2_normal(point3D, lc_window_sz):
    b, c, h, w = point3D.size()
    assert (c == 3)

    dx = point3D[:, :, :, lc_window_sz:] - point3D[:, :, :, :-lc_window_sz]
    dy = point3D[:, :, :-lc_window_sz,:] - point3D[:, :, lc_window_sz:, :]

    dx = dx[:, :, lc_window_sz:, :]
    dy = dy[:, :, :, :-lc_window_sz]
    assert (dx.size() == dy.size())

    normal = torch.cross(dx, dy, dim=1)
    assert (normal.size() == dx.size())

    normal = F.normalize(normal, dim=1, p=2)
    return -normal



def compute_normal_errors(pred, gt, mask):
    """Computation of error metrics between predicted and ground truth norms
    """ 
    pred = F.normalize(pred, p=2, dim=1).data.cpu()
    gt = F.normalize(gt, p=2, dim=1).data.cpu()    
    cos_theta = torch.sum(pred*gt, dim=1).numpy()[0]
    mask = mask.data.cpu().numpy()[0,0]
    valid_entries = cos_theta[np.where(mask)].clip(-1, 1)
    angle = np.arccos(valid_entries) / np.pi * 180
    
    mean = angle.mean()
    rmse = (angle**2).mean()**0.5
    a1 = (angle<11.25).mean()
    a2 = (angle<22.5).mean()
    a3 = (angle<30).mean()
    a4 = (angle<40).mean()

    return mean, rmse, a1, a2, a3, a4
 

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
    lg10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, lg10, a1, a2, a3


def prepare_model_for_test(opt):
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, 
            scales=range(1), 
            upsample_mode='bilinear'
    )

    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda().eval()
    depth_decoder.cuda().eval()
    
    return encoder, depth_decoder, encoder_dict['height'], encoder_dict['width']



def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    lc_window_sz = 1

    ratios = []
    normal_errors = []
        
    encoder, depth_decoder, thisH, thisW = prepare_model_for_test(opt)
    backproject_depth = BackprojectDepth(1, thisH, thisW)

    filenames = readlines('./splits/nyu_test.txt')
    dataset = datasets.NYUTestDataset(
            opt.data_path,
            filenames,
            thisH, thisW,
    )
    
    dataloader = DataLoader(
            dataset, 1, shuffle=False, 
            num_workers=opt.num_workers
    )
    print("-> Computing predictions with size {}x{}".format(thisH, thisW))

    with torch.no_grad():
        for ind, (data, _, gt_norm, gt_norm_mask, K, K_inv) in enumerate(tqdm(dataloader)):
            input_color = data.cuda()
            if opt.post_process:
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
            output = depth_decoder(encoder(input_color))

            pred_disp, _ = disp_to_depth(
                    output[("disp", 0)], 
                    opt.min_depth, 
                    opt.max_depth
            )
            pred_disp = pred_disp.data.cpu() 

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(
                        pred_disp[:N], torch.flip(pred_disp[N:], [3]) 
                )
            pred_depth = 1 / pred_disp

            cam_points = backproject_depth(pred_depth, K_inv)
            cam_points = cam_points[:, :3, ...].view(1, 3, thisH, thisW)
            normal = depth_2_normal(cam_points, lc_window_sz)

            normal = F.pad(normal, (0, lc_window_sz, 0, lc_window_sz), mode='replicate')
            normal = F.interpolate(normal, (gt_norm.shape[2], gt_norm.shape[3])) 
            normal_errors.append(compute_normal_errors(normal, gt_norm, gt_norm_mask))

    mean_normal_errors = np.array(normal_errors).mean(0)
    
    print("\n  " + ("{:>8} | " * 6).format("mean", "rmse", "a1", "a2", "a3", "a4"))
    print(("&{: 8.3f}  " * 6).format(*mean_normal_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
