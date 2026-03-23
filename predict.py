import logging
import os
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import scipy.misc
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from medpy.metric import hd95
import csv
from torch.utils.tensorboard import SummaryWriter

from utils.meter import *
from utils.helpers import *
# from utils.visualize import visualize_heads
# from visualizer import get_local

# get_local.activate()

cudnn.benchmark = True

def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)
    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 1.0
            # follow ACN and SMU-Net
            # return 373.12866
            # follow nnUNet
    elif num_pred == 0 and num_ref != 0:
        return 1.0
        # follow ACN and SMU-Net
        # return 373.12866
        # follow in nnUNet
    else:
        return hd95(pred, ref, (1, 1, 1))

def cal_hd95(output, target):
     # whole tumor
    mask_gt = (target != 0).astype(int)
    mask_pred = (output != 0).astype(int)
    hd95_whole = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # tumor core
    mask_gt = ((target == 1) | (target ==3)).astype(int)
    mask_pred = ((output == 1) | (output ==3)).astype(int)
    hd95_core = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # enhancing
    mask_gt = (target == 3).astype(int)
    mask_pred = (output == 3).astype(int)
    hd95_enh = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    mask_gt = (target == 3).astype(int)
    if np.sum((output == 3).astype(int)) < 500:
       mask_pred = (output == 3).astype(int) * 0
    else:
       mask_pred = (output == 3).astype(int)
    hd95_enhpro = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    return (hd95_whole, hd95_core, hd95_enh, hd95_enhpro)

def softmax_output_dice_class4(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3 
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def compute_dice(pred, target, eps=1e-8):
    """Calculate the Dice Coefficient."""
    intersection = torch.sum(pred * target, dim=(1,2,3))
    union = torch.sum(pred, dim=(1,2,3)) + torch.sum(target, dim=(1,2,3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.cpu().numpy()

def compute_hd95(pred, target):
    """Calculate the 95% Hausdorff Distance (HD95)."""
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    return hd95(pred_np, target_np)

def test_dice_hd95_softmax(
        test_loader,
        model,
        feature_mask=None,
        csv_name=None, 
        patch_size=(128,128,128),
        save_image=False,
        mask_name='t2'
    ):
    model.eval()
    vals_dice_evaluation = AverageMeter()
    vals_hd95_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    patch_size_h, patch_size_w, patch_size_z = patch_size
    one_tensor = torch.ones(1, patch_size_h, patch_size_w, patch_size_z).float().cuda()

    num_cls = 4
    class_evaluation = ('whole', 'core', 'enhancing', 'enhancing_postpro')
    class_separate = ('ncr_net', 'edema', 'enhancing')

    for i, data in enumerate(test_loader):
        if i+1 in [75,24,26,9,32,3,29,16,25,71]:
            # Load data
            inputs = data[0].float().cuda()
            B, _, H, W, Z = inputs.size()
            target = data[1].long().cuda()
            names = data[-1]
            if feature_mask is not None:
                mask = torch.from_numpy(np.array(feature_mask))
                mask = torch.unsqueeze(mask, dim=0).repeat(B, 1)
            else:
                mask = data[2]
            mask = mask.bool().cuda()

            # Get sliding window indices
            h_idx_list = list(np.arange(0, H - patch_size_h + 1, int(patch_size_h * 0.5)))
            w_idx_list = list(np.arange(0, W - patch_size_w + 1, int(patch_size_w * 0.5)))
            z_idx_list = list(np.arange(0, Z - patch_size_z + 1, int(patch_size_z * 0.5)))

            h_idx_list.append(H - patch_size_h)
            w_idx_list.append(W - patch_size_w)
            z_idx_list.append(Z - patch_size_z)

            # Compute weight matrix
            weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        weight1[:, :, h:h+patch_size_h, w:w+patch_size_w, z:z+patch_size_z] += one_tensor
            weight = weight1.repeat(B, num_cls, 1, 1, 1)

            # Sliding window inference
            pred = torch.zeros(B, num_cls, H, W, Z).float().cuda()
            model.module.is_training = False
            
            with torch.no_grad():        
                for h in h_idx_list:
                    for w in w_idx_list:
                        for z in z_idx_list:
                            x_input = inputs[:, :, h:h+patch_size_h, w:w+patch_size_w, z:z+patch_size_z]
                            pred_part = model(x_input, mask)
                            pred[:, :, h:h+patch_size_h, w:w+patch_size_w, z:z+patch_size_z] += pred_part
                            
            pred = pred / weight
            pred = pred[:, :, :H, :W, :Z]
            pred = torch.argmax(pred, dim=1)
            target = torch.argmax(target, dim=1)

            scores_dice_separate, scores_dice_evaluation = softmax_output_dice_class4(pred, target)

            for k, name in enumerate(names):
                msg = f'Subject {i+1}/{len(test_loader)}, {k+1}/{B}: {name}, '
                vals_separate.update(scores_dice_separate[k])
                vals_dice_evaluation.update(scores_dice_evaluation[k])
                scores_hd95 = np.array(cal_hd95(pred[k].cpu().numpy(), target[k].cpu().numpy()))
                vals_hd95_evaluation.update(scores_hd95)
                
                msg += 'DSC: ' + ', '.join([f'{k}: {v:.4f}' for k, v in zip(class_evaluation, scores_dice_evaluation[k])])
                msg += ', HD95: ' + ', '.join([f'{k}: {v:.4f}' for k, v in zip(class_evaluation, scores_hd95)])
                print_log(msg)

                # Save to CSV if specified
                if csv_name:
                    with open(csv_name, "a+") as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerow([name, scores_dice_evaluation[k][0], scores_dice_evaluation[k][1], scores_dice_evaluation[k][2], scores_dice_evaluation[k][3],
                                            scores_hd95[0], scores_hd95[1], scores_hd95[2], scores_hd95[3]])

                if save_image:
                    save_dir = os.path.dirname(csv_name)
                    saved_image_dir = os.path.join(save_dir, 'saved_image')

                    os.makedirs(saved_image_dir, exist_ok=True)

                    np.save(os.path.join(saved_image_dir, f'{name}_{mask_name}_img.npy'), inputs[k].cpu().numpy())
                    np.save(os.path.join(saved_image_dir, f'{name}_{mask_name}_gt.npy'), target[k].cpu().numpy())
                    np.save(os.path.join(saved_image_dir, f'{name}_{mask_name}_pred.npy'), pred[k].cpu().numpy())

            
    # Final average scores
    msg = 'Average scores: '
    msg += ', '.join([f'{k}: {v:.4f}' for k, v in zip(class_evaluation, vals_dice_evaluation.avg)])
    msg += ', '.join([f'{k}: {v:.4f}' for k, v in zip(class_evaluation, vals_hd95_evaluation.avg)])
    print_log(msg)

    if csv_name:
        with open(csv_name, "a+") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Average', vals_dice_evaluation.avg[0], vals_dice_evaluation.avg[1], vals_dice_evaluation.avg[2], vals_dice_evaluation.avg[3],
                                vals_hd95_evaluation.avg[0], vals_hd95_evaluation.avg[1], vals_hd95_evaluation.avg[2], vals_hd95_evaluation.avg[3]])

    model.train()
    return vals_dice_evaluation.avg, vals_hd95_evaluation.avg

def test_dice_hd95_softmax_cc(
        test_loader,
        model,
        feature_mask=None,
        csv_name=None, 
        patch_size=(128,128,128),
        save_image=False,
        mask_name='t2'
    ):
    model.eval()
    vals_dice_evaluation = AverageMeter()
    vals_hd95_evaluation = AverageMeter()
    patch_size_h, patch_size_w, patch_size_z = patch_size
    one_tensor = torch.ones(1, patch_size_h, patch_size_w, patch_size_z).float().cuda()

    num_cls = 2
    for i, data in enumerate(test_loader):
        # Load data
        inputs = data[0].float().cuda()
        B, _, H, W, Z = inputs.size()
        target = data[1].long().cuda()
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(B, 1)
        else:
            mask = data[2]
        mask = mask.bool().cuda()

        # Get sliding window indices
        h_idx_list = list(np.arange(0, H - patch_size_h + 1, int(patch_size_h * 0.5)))
        w_idx_list = list(np.arange(0, W - patch_size_w + 1, int(patch_size_w * 0.5)))
        z_idx_list = list(np.arange(0, Z - patch_size_z + 1, int(patch_size_z * 0.5)))

        h_idx_list.append(H - patch_size_h)
        w_idx_list.append(W - patch_size_w)
        z_idx_list.append(Z - patch_size_z)

        # Compute weight matrix
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + patch_size_h, w:w + patch_size_w, z:z + patch_size_z] += one_tensor
        weight = weight1.repeat(B, num_cls, 1, 1, 1)

        # Sliding window inference
        pred = torch.zeros(B, num_cls, H, W, Z).float().cuda()
        model.module.is_training = False
        
        with torch.no_grad():
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        x_input = inputs[:, :, h:h + patch_size_h, w:w + patch_size_w, z:z + patch_size_z]
                        pred_part = model(x_input, mask)
                        pred[:, :, h:h + patch_size_h, w:w + patch_size_w, z:z + patch_size_z] += pred_part

        pred = pred / weight
        pred = pred[:, :, :H, :W, :Z]
        pred = torch.argmax(pred, dim=1)
        target = torch.argmax(target, dim=1)

        scores_dice = compute_dice(pred, target)

        for k, name in enumerate(names):
            vals_dice_evaluation.update(scores_dice[k])
            scores_hd95 = compute_hd95(pred[k], target[k])
            vals_hd95_evaluation.update(scores_hd95)

            msg = f'Subject {i + 1}/{len(test_loader)}, {name:>20}: DSC: {scores_dice[k]:.4f}, HD95: {scores_hd95:.4f}'
            print_log(msg)

            # Save to CSV if specified
            if csv_name:
                with open(csv_name, "a+", newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow([name, scores_dice[k], scores_hd95])
            
            if save_image:
                save_dir = os.path.dirname(csv_name)
                saved_image_dir = os.path.join(save_dir, 'saved_image')

                os.makedirs(saved_image_dir, exist_ok=True)

                np.save(os.path.join(saved_image_dir, f'{name}_{mask_name}_img.npy'), inputs[k].cpu().numpy())
                np.save(os.path.join(saved_image_dir, f'{name}_{mask_name}_gt.npy'), target[k].cpu().numpy())
                np.save(os.path.join(saved_image_dir, f'{name}_{mask_name}_pred.npy'), pred[k].cpu().numpy())

    # Final average scores
    avg_dice = vals_dice_evaluation.avg
    avg_hd95 = vals_hd95_evaluation.avg
    avg_msg = f'Average DSC: {avg_dice:.4f}, Average HD95: {avg_hd95:.4f}'
    print_log(avg_msg)

    if csv_name:
        with open(csv_name, "a+") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Average', avg_dice, avg_hd95])


    model.train()
    return avg_dice, avg_hd95



