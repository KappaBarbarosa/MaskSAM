import numpy as np
import torch
import torch.nn.functional as F
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from medpy import metric
import cv2
import monai
from einops import repeat

def dice_coef(y_true, y_pred, smooth=1):
    # print(y_pred.shape, y_true.shape)
    intersection = torch.sum(y_true * y_pred,axis=(-1,-2))
    union = torch.sum(y_true, axis=(-1,-2)) + torch.sum(y_pred, axis=(-1,-2))
    dice = ((2. * intersection + smooth)/(union + smooth)).mean()
    # print(dice)
    return dice

def iou_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred),axis=(-1,-2))
    union = torch.sum(y_true,axis=(-1,-2))+torch.sum(y_pred,axis=(-1,-2))-intersection
    iou = ((intersection + smooth) / (union + smooth)).mean()
    return iou

def running_stats(y_true, y_pred, smooth = 1):
    intersection = torch.sum(y_true * y_pred,axis=(-1,-2))
    union = torch.sum(y_true, axis=(-1,-2)) + torch.sum(y_pred, axis=(-1,-2))
    return intersection, union

def dice_collated(running_intersection, running_union, smooth =1):
    if len(running_intersection.size())==2:
        dice = (torch.mean((2. * running_intersection + smooth)/(running_union + smooth),dim=1)).sum()
    else:
        dice = ((2. * running_intersection + smooth)/(running_union + smooth)).sum()
    return dice

def dice_batchwise(running_intersection, running_union, smooth=1):
    dice = ((2. * running_intersection + smooth)/(running_union + smooth))
    return dice


def dice_coefficient(prediction, target):
    intersection = torch.sum(prediction * target)
    union = torch.sum(prediction) + torch.sum(target)
    dice_coeff = (2.0 * intersection) / (union + 1e-8)  # Adding a small epsilon to avoid division by zero
    return dice_coeff


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: dice_loss.py
# description:
# implementation of dice loss for NLP tasks.

def thresholding(batch_mask):
    return (batch_mask > 0.5).astype(int)

def calculate_IOU(mask1, mask2):
    intersection = np.logical_and(np.array(mask1), np.array(mask2))
    union = np.logical_or(np.array(mask1), np.array(mask2))
    if np.sum(union) != 0:
        iou = np.sum(intersection) / np.sum(union)
    elif np.sum(intersection) == 0:
        iou = 1
    else:
        iou = 0
    return iou 

def calculate_metric_percase(pred, gt):
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        print('normal')
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        print('all zero')
        return 0, 0
def calculate_metric_perslice(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0
def data_transform(image,desired_size=1024):
        image = cv2.resize(image, (desired_size, desired_size), interpolation=cv2.INTER_CUBIC)
        image = torch.as_tensor(image)
        if image.dim() == 2: # for gray image
            image = image.unsqueeze(0)
            image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        else: # RGB image
            image = image.permute(2, 0, 1)
        return image
