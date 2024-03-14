'''
@copyright ziqi-jin
You can create custom loss function in this file, then import the created loss in ./__init__.py and add the loss into AVAI_LOSS
'''
import torch
import torch.nn.functional as F
import torch.nn as nn


# example
class CustormLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, y):
        pass


# 定义Dice Loss作为nn.Module
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, labels):
        # 将labels转换为one-hot编码
        # labels_one_hot = F.one_hot(labels, num_classes=2).permute(0, 3, 1, 2).float()

        # 计算smooth项，避免除零错误
        pred = torch.argmax(pred, dim=1).float()
        smooth = 1e-6

        # 计算交集
        intersection = torch.sum(pred * labels, dim=(1, 2))

        # 计算各自的和
        pred_sum = torch.sum(pred, dim=(1, 2))
        labels_sum = torch.sum(labels, dim=(1, 2))

        # 计算Dice系数
        dice_coeff = (2. * intersection + smooth) / (pred_sum + labels_sum + smooth)

        # 计算Dice Loss
        dice_loss = 1 - dice_coeff

        # 计算批量的平均Dice Loss
        loss = torch.mean(dice_loss)

        return loss

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()