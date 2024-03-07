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

def dice_loss(predict, target, labels, label_weight):
    assert predict.size() == target.size(), "the size of predict and target must be equal."
    epsilon = 1e-5
    num = predict.size(0)
    
    pre = torch.sigmoid(predict).view(num, -1)
    tar = target.view(num, -1)
    
    intersection = (pre * tar).sum(-1).sum()
    union = (pre + tar).sum(-1).sum()
    
    score = 1 -  (2.0 * intersection + epsilon) / (union + epsilon)
    
    return 'Dice loss', score

def dice_coefficient(prediction, target):
    intersection = torch.sum(prediction * target)
    union = torch.sum(prediction) + torch.sum(target)
    dice_coeff = (2.0 * intersection) / (union + 1e-8)  # Adding a small epsilon to avoid division by zero
    return dice_coeff

def weighted_dice_loss(y_pred, y_true, label, label_penalty_weight):
    loss = 0

    for predict, ground_truth, label in zip(y_pred, y_true, label):
        pt = predict.unsqueeze(0)
        gt = ground_truth.unsqueeze(0)
        loss += label_penalty_weight[label] * (1 - dice_coefficient(pt, gt))
    return 'Dice loss', loss

def weighted_BCE_loss(y_pred, y_true, label, label_penalty_weight):
    # BCEWithLogitsLoss combines sigmoid activation and BCE loss
    loss = 0

    for predict, ground_truth, label in zip(y_pred, y_true, label):
        pt = predict.unsqueeze(0)
        gt = ground_truth.unsqueeze(0)
        loss += label_penalty_weight[label] * torch.nn.BCEWithLogitsLoss()(pt, gt)
        
    return 'BCE loss', loss
#     return 'BCE loss', torch.nn.BCEWithLogitsLoss()(y_pred, y_true)

def BCE_loss(y_pred, y_true, label=None, label_penalty_weight=None):
    return 'BCE loss', torch.nn.BCEWithLogitsLoss()(y_pred, y_true)
def monai_diceloss(y_pred, y_true,label,label_weight):
    return 'monai Dice loss',monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")(y_pred, y_true)
def monai_focal_loss(y_pred, y_true,label,label_weight):
    return 'monai Focal loss',monai.losses.FocalLoss(reduction="none", gamma=3.5)(y_pred, y_true)



def weighted_ce_loss(y_pred, y_true, alpha=64, smooth=1):
    weight1 = torch.sum(y_true==1,dim=(-1,-2))+smooth
    weight0 = torch.sum(y_true==0, dim=(-1,-2))+smooth
    multiplier_1 = weight0/(weight1*alpha)
    multiplier_1 = multiplier_1.view(-1,1,1)


    loss = -torch.mean(torch.mean((multiplier_1*y_true*torch.log(y_pred)) + (1-y_true)*(torch.log(1-y_pred)),dim=(-1,-2)))
    return 'weighted CE loss', loss

def focal_loss(y_pred, y_true, alpha_def=0.75, gamma=3):
    # print('going back to the default value of alpha')
    alpha = alpha_def
    ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
    assert (ce_loss>=0).all()
    p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    # 1/0
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    loss = alpha_t * loss
    loss = torch.sum(loss, dim=(-1,-2))
    return 'focal loss', loss.mean()

def binary_focal_loss(y_pred, y_true, labels, label_weight):
    return 'focal loss',BinaryFocalLoss()(y_pred, y_true, labels, label_weight)

def weighted_focal_loss(y_pred, y_true,labels,label_weight):
    return 'focal loss',WeightedBinaryFocalLoss()(y_pred, y_true,labels,label_weight)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=3, ignore_index=None, reduction='none', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target, labels, label_weight):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        return loss
    
class WeightedBinaryFocalLoss(nn.Module):
    def __init__(self, alpha=3, gamma=3.5, reduction='none', **kwargs):
        super(WeightedBinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            print(reduction)
        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target, labels, label_weight):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)
    
        weight_for_loss = label_weight.gather(0, labels).to(output.device)
        
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        
        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)
        
        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)
        
        loss = pos_loss + neg_loss
        loss = weight_for_loss.unsqueeze(1).unsqueeze(1) * loss 
        if self.reduction == 'mean':
            loss = loss.mean(),
        elif self.reduction == 'sum':
            loss = loss.sum()            
        return loss   
    
def binary_dice_loss(y_pred, y_true):
    return 'Dice loss',BinaryDiceLoss()(y_pred, y_true)

class BinaryDiceLoss(nn.Module):
    def __init__(self, ignore_index=None, reduction='none', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1  # suggest set a large number when target area is large,like '10|100'
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.batch_dice = False  # treat a large map when True
        if 'batch_loss' in kwargs.keys():
            self.batch_dice = kwargs['batch_loss']

    def forward(self, output, target, use_sigmoid=True):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        if use_sigmoid:
            output = torch.sigmoid(output)

        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.mul(validmask)  # can not use inplace for bp
            target = target.float().mul(validmask)

        dim0 = output.shape[0]
        if self.batch_dice:
            dim0 = 1

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
