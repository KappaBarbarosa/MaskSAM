import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import monai
from Utils.Score import dice_coefficient
import numpy as np
def compute_mask(output, target, threshold):
    condition = ((output > threshold).sum(dim=[1, 2]) == 0) & (target.sum(dim=[1, 2]) == 0)
    mask = condition.view(-1, 1, 1).float()
    return mask
    
def dice_loss(predict, target, labels, label_weight):
    assert predict.size() == target.size(), "the size of predict and target must be equal."
    epsilon = 1e-5
    num = predict.size(0)
    
    pre = torch.sigmoid(predict).view(num, -1)
    tar = target.view(num, -1)

    
    
    intersection = (pre * tar).sum(-1).sum()
    union = (pre + tar).sum(-1).sum()
    
    score = 1 -  (2.0 * intersection + epsilon) / (union + epsilon)
    
    special_loss = torch.zeros_like(score)

    # Compute the mask for special condition
    mask = (((pre > 0.5).sum() == 0) & (target.sum() == 0)).float()

    # Combine the losses
    score = mask * special_loss + (1 - mask) * score

    return 'Dice loss', score
def adjust_diceloss(y_pred,y_true, labels, label_weight):
    return 'Dice loss', BinaryAdjustDiceLoss()(y_pred, y_true,labels)
class BinaryAdjustDiceLoss(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)

    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    """
    def __init__(self,
                 smooth: Optional[float] = 1e-4,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 ohem_ratios = np.round(np.array([0.31685527, 0.32872596, 0.32553957, 0.11497731, 0.70076923,
       0.36734694, 1.21987952, 0.24143739]),3),
                 alpha: float = 2,
                 reduction: Optional[str] = "none",
                 index_label_position=True) -> None:
        super(BinaryAdjustDiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratios = ohem_ratios
        self.alpha = alpha
        self.index_label_position = index_label_position

    def forward(self, input: Tensor, target: Tensor, label: Optional[Tensor] = None) -> Tensor:
        loss = self._binary_class(input, target, label=label)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            loss = 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            loss = 1 - ((2 * interection + self.smooth) /
                        (torch.sum(torch.square(flat_input, ), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))

        return loss

    def _binary_class(self, input, target, label, mask=None):
        batchsize = input.size(0)
        flat_input = input.view(batchsize, -1)
        flat_target = target.view(batchsize, -1).float()

        flat_input = torch.sigmoid(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(flat_target)

        for i in range(batchsize):
            if self.ohem_ratios[label[i]] > 0:
                old_input = flat_input[i].detach()
                pos_example = flat_target[i].detach() > 0.5
                neg_example = flat_target[i] <= 0.5
                mask_neg_num = mask[i] <= 0.5

                pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
                neg_num = neg_example.sum()
                keep_num = min(int(pos_num * self.ohem_ratios[label[i]]), neg_num)

                neg_scores = torch.masked_select(old_input, neg_example.bool())
                neg_scores_sort, _ = torch.sort(neg_scores, )
                threshold = neg_scores_sort[-keep_num+1]
                cond = (old_input> threshold) | pos_example.view(-1)
                ohem_mask = torch.where(cond, 1, 0)
                flat_input[i] = flat_input[i] * ohem_mask
                flat_target[i] = flat_target[i] * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)


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
        print('output')
        print(prob)
        print('target')
        print(target.sum())
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        print('loss')
        print(loss)
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
    
def binary_dice_loss(y_pred, y_true, labels, label_weight):
    return 'Dice loss',BinaryDiceLoss()(y_pred, y_true)

class BinaryDiceLoss(nn.Module):
    def __init__(self, ignore_index=None,use_filter=True, reduction='none', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1  # suggest set a large number when target area is large,like '10|100'
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.use_filter = use_filter

    def forward(self, output, target, use_sigmoid=True):
        output = torch.sigmoid(output)
#         print('output')
#         print(output)
#         print('target')
#         print(target)
        mask = compute_mask (output.detach(),target,0.05)        
        output = output*(1-mask)

        dim0 = output.shape[0]

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth

        loss = 1 - (num / den)
#         print('loss')
#         print(loss)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
