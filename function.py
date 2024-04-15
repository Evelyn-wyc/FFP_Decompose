import torch
import cv2
import numpy as np
import os

def log10(t):
    numerator = torch.log(t)
    denominator = torch.log(torch.FloatTensor([10.])).cuda()
    return numerator / denominator

def psnr_error(gen_frames, gt_frames):
    # gen_frames means the output of the generator
    # gt_frames means the ground truth
    shape = list(gen_frames.shape)
    num_pixels = (shape[0] * shape[1])
    gt_frames = (gt_frames + 1.0) / 2.0
    gen_frames = (gen_frames + 1.0) / 2.0
    square_diff = (gt_frames - gen_frames) ** 2
    batch_errors = 10 * log10(1. / ((1. / num_pixels) * torch.sum(square_diff, [1, 2, 3])))
    return torch.mean(batch_errors)

# def log10(t):
#     numerator = np.log(t)
#     denominator = np.log(10)
#     return numerator / denominator
#
# def psnr_error(gen_frames, gt_frames):
#     shape = list(gen_frames.shape)
#     num_pixels = (shape[0] * shape[1])
#     gt_frames = (gt_frames + 1.0) / 2.0
#     gen_frames = (gen_frames + 1.0) / 2.0
#     square_diff = (gt_frames - gen_frames) ** 2
#     batch_errors = 10 * log10(1. / ((1. / num_pixels) * np.sum(square_diff)))
#     return np.mean(batch_errors)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# 计算交并比
def dice_coefficient(gt, anomaly):
    intersection = np.sum(gt * anomaly)
    union = np.sum(gt) + np.sum(anomaly)
    if union == 0:
        dice = 1
    else:
        dice = 2 * intersection / union
    return dice


