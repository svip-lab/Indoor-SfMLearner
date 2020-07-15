from __future__ import absolute_import, division, print_function

import os, sys
sys.path.append(os.getcwd())
import numpy as np

import torch
import torch.nn.functional as F
import datasets
import networks

from tqdm import tqdm
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions


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


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

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
    errors = []
    ratios = []

    encoder, depth_decoder, thisH, thisW = prepare_model_for_test(opt)

    filenames = readlines('./splits/scannet_test_depth.txt')
    dataset = datasets.ScannetTestDepthDataset(
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
        for ind, (data, gt_depth) in enumerate(tqdm(dataloader)):
            input_color = data.cuda()
            if opt.post_process:
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
            output = depth_decoder(encoder(input_color))
             
            disp = output['disp', 0]
            disp = F.interpolate(disp, (gt_depth.shape[2], gt_depth.shape[3]))
            pred_disp, _ = disp_to_depth(disp, opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu().squeeze(1).numpy()
            
            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(
                        pred_disp[:N], pred_disp[N:, :, ::-1]
                )
            pred_depth = 1 / pred_disp
           
            pred_depth = pred_depth[0]
            gt_depth = gt_depth.data.numpy()[0,0]

            mask = gt_depth > 0
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
            
            pred_depth[pred_depth < opt.min_depth] = opt.min_depth
            pred_depth[pred_depth > opt.max_depth] = opt.max_depth
            errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "lg10", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
