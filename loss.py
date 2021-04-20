#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class IoULoss(nn.Module):
    """
    https://stats.stackexchange.com/questions/321460/dice-coefficient-loss-function-vs-cross-entropy 
    """
    def __init__(self, n_classes, ignore_lb=255):
        super(IoULoss, self).__init__()

        self.n_classes = n_classes
        self.ignore_lb = ignore_lb

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)

        labels[labels == self.ignore_lb] = self.n_classes
        true = torch.nn.functional.one_hot(labels, self.n_classes + 1).permute(0, 3, 1, 2).float()
        true = true[:, :self.n_classes, :, :]
        
        inter = true * probs
        cardinality = probs + true
        union = cardinality - inter

        iou = torch.sum(inter) / (torch.sum(union) + 1e-7)
        iou_loss = 1 - iou
        
        return iou_loss
    

class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes, ignore_lb=255):
        super(DiceLoss, self).__init__()
        
        self.n_classes = n_classes
        self.ignore_lb = ignore_lb

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)

        labels[labels == self.ignore_lb] = self.n_classes
        true = torch.nn.functional.one_hot(labels, self.n_classes + 1).permute(0, 3, 1, 2).float()
        true = true[:, :self.n_classes, :, :]
        
        inter = torch.sum(true * probs)
        cardinality = torch.sum(probs + true)

        dice = inter / (cardinality + 1e-7)
        dice_loss = 1 - dice
        
        return dice_loss


class OHIoULoss(nn.Module):
    """
    https://stats.stackexchange.com/questions/321460/dice-coefficient-loss-function-vs-cross-entropy 
    """
    def __init__(self, n_classes, n_min, thresh=0.3, ignore_lb=255, *args, **kwargs):
        super(OHIoULoss, self).__init__()

        self.n_classes = n_classes
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.thresh = thresh

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)

        labels[labels == self.ignore_lb] = self.n_classes
        true = torch.nn.functional.one_hot(labels, self.n_classes + 1).permute(0, 3, 1, 2).float()
        true = true[:, :self.n_classes, :, :]
        
        inter = true * probs
        cardinality = probs + true
        union = cardinality - inter

        iou = inter / (union + 1e-7)
        loss = torch.sum(1 - iou, dim=0)

        # loss, _ = torch.sort(loss.view(-1), descending=True)
        # if loss[self.n_min] > self.thresh:
        #     loss = loss[loss>self.thresh]
        # else:
        #     loss = loss[:self.n_min]

        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    OHIoULoss = OHIoULoss(2, thresh=0.7, n_min=16*20*20//16).cuda()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

        loss1 = criteria1(logits1, lbs)
        print(loss.detach().cpu())
    