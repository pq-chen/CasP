import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from skimage import io
import random
import sys
import cv2
import matplotlib.pyplot as plt 
from .torch_geometry import get_perspective_transform
# from pytorch3d.transforms import euler_angles_to_matrix
from .torch_geometry import euler_angles_to_matrix, get_perspective_transform

import time
import math

def show_overlap(img1, img2, H, show=True, g_u=0, g_v=0):
    # Display overlap from img1 to img2 (np.array)
    image1 = img1.copy()
    image0 = img2.copy()
    h, w = image0.shape[0], image0.shape[1]
    h_, w_ = image1.shape[0], image1.shape[1]

    result = cv2.warpPerspective(image1, H, (w + w + image1.shape[1], h))  # result placing image1 warped image on the left
    mask_temp = result[:, 0:w] > 1
    temp2 = result[:, 0:w, :].copy()
    frame = image0.astype(np.uint8)
    roi = frame[mask_temp]
    frame[mask_temp] = (0.5 * temp2.astype(np.uint8)[mask_temp] + 0.5 * roi).astype(np.uint8)
    result[0:image0.shape[0], image1.shape[1]:image0.shape[1] + image1.shape[1]] = frame  # result placing overlap in the middle
    result[0:h, image1.shape[1] + w:w + w + image1.shape[1]] = image0  # result placing image0 on the right
    pts = np.float32([[0, 0], [0, h_], [w_, h_], [w_, 0]]).reshape(-1, 1, 2)
    center = np.float32([w_ / 2, h_ / 2]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    dst_center = cv2.perspectiveTransform(center, H).reshape(-1, 2)
    # Adding offsets
    for i in range(4):
        dst[i][0] += w_ + w
    dst_center[0][0] += w_ + w
    cv2.polylines(result, [np.int32(dst)], True, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(result, (int(dst_center[0][0]), int(dst_center[0][1])), 15, (0, 255, 0), 1)
    cv2.circle(temp2, (int(dst_center[0][0] - w_ - w), int(dst_center[0][1])), 15, (0, 255, 0), 1)
    if show:
        plt.subplot(1, 3, 1)
        plt.imshow(temp2.astype(np.uint8))
        plt.subplot(1, 3, 2)
        plt.imshow(result[:, w_:w_ + w].astype(np.uint8))
        plt.subplot(1, 3, 3)
        if g_u != 0 and g_v != 0:
            result = draw_markers(np.ascontiguousarray(result), [g_u + w + w_, g_v], size=4, thickness=2, color=(255, 0, 0), shape=2)
        plt.imshow(result[:, w + w_:].astype(np.uint8))
    return result


def get_homograpy(four_point, sz, k = 1):
    """
    four_point: four corner flow
    sz: image size
    k: scale
    Shape:
        - Input: :four_point:`(B, 2, 2, 2)` and :sz:`(B, C, H, W)`
        - Output: :math:`(B, 3, 3)`
    """
    N,_,h,w = sz
    h, w = h//k, w//k
    four_point = four_point / k 
    four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
    four_point_org[:, 0, 0] = torch.Tensor([0, 0])
    four_point_org[:, 0, 1] = torch.Tensor([w-1, 0])
    four_point_org[:, 1, 0] = torch.Tensor([0, h-1])
    four_point_org[:, 1, 1] = torch.Tensor([w -1, h-1])

    four_point_org = four_point_org.unsqueeze(0)
    four_point_org = four_point_org.repeat(N, 1, 1, 1)
    four_point_new = torch.autograd.Variable(four_point_org) + four_point
    four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
    four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
    # H = tgm.get_perspective_transform(four_point_org, four_point_new)
    H = get_perspective_transform(four_point_org, four_point_new)
    return H


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    # [1024, 1, 32, 32]  [1024, 9, 9, 2] https://www.jb51.net/article/273930.htm
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)  # [1024, 9, 9, 2]
    # img = F.grid_sample(img, grid, align_corners=True) # [1024, 1, 9, 9]
    # img = F.grid_sample(img, grid, align_corners=False)
    try:
        img = F.grid_sample(img, grid, align_corners=True) # [1024, 1, 9, 9]
    except Exception as e:
        # import pdb;pdb.set_trace()
        with torch.backends.cudnn.flags(enabled=False):
            img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd),indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float() 

    return coords[None].expand(batch, -1, -1, -1)

def save_img(img, path):
    npimg = img.detach().cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.astype(np.uint8)
    io.imsave(path, npimg)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_params(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())  
        Total_params += mulValue  
        if param.requires_grad:
            Trainable_params += mulValue  
        else:
            NonTrainable_params += mulValue 

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask

def warp_coor(x, coor):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    coor: [B, 2, H, W] coor2 after H
    """
    B, C, H, W = x.size()
    # mesh grid
    vgrid = torch.autograd.Variable(coor)

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask

class TextColors:
    RED = '31'
    GREEN = '32'
    YELLOW = '33'
    BLUE = '34'
    MAGENTA = '35'
    CYAN = '36'
    WHITE = '37'

def print_colored(text, color=TextColors.RED):
    print(f"\033[{color}m{text}\033[0m")

