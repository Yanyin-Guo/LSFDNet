import os
import torchvision
import random
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize, Grayscale
from scripts.imagecrop import vi_FusionlabelCrop, vi_FusionCenterCrop, FusionRandomCrop, FusionCenterCrop

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    return Compose([
        FusionRandomCrop(crop_size),
    ])

def train_centerhr_transform(crop_size):
    return Compose([
        FusionCenterCrop(crop_size),
    ])

def train_vi_dethr_transform(crop_size):
    return Compose([
        vi_FusionlabelCrop(crop_size),
    ])

def train_vi_centerhr_transform(crop_size):
    return Compose([
        vi_FusionCenterCrop(crop_size),
    ])

def test_size(size):
    x = size[0]-size[0]%16
    y = size[1]-size[1]%16
    return [size[0],size[1]],[x,y]

def RGB2YCrCb(input_im):
    if len(input_im.shape) == 3:  # 输入为 (H, W, C)  
        input_im = input_im.unsqueeze(0)
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    if len(input_im.shape) == 3:  # 输入为 (H, W, C)  
        input_im = input_im.unsqueeze(0)
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out