import os
import cv2
import math
import time
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
from scipy import ndimage
import torch.nn.functional as F
from sklearn.metrics import *
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

class Evaluator:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.MAE = list()
        self.Recall = list()
        self.Precision = list()
        self.Accuracy = list()
        self.Dice = list()       
        self.IoU_polyp = list()

    
    def evaluate(self, pred, gt):
        
        pred_binary = (pred >= 0.45).float().cuda()
        pred_binary_inverse = (pred_binary == 0).float().cuda()

        gt_binary = (gt >= 0.5).float().cuda()
        gt_binary_inverse = (gt_binary == 0).float().cuda()
        
        MAE = torch.abs(pred_binary - gt_binary).mean().cuda(0)
        TP = pred_binary.mul(gt_binary).sum().cuda(0)
        FP = pred_binary.mul(gt_binary_inverse).sum().cuda(0)
        TN = pred_binary_inverse.mul(gt_binary_inverse).sum().cuda(0)
        FN = pred_binary_inverse.mul(gt_binary).sum().cuda(0)

        if TP.item() == 0:
            TP = torch.Tensor([1]).cuda(0)
        # recall
        Recall = TP / (TP + FN)
        # Precision or positive predictive value
        Precision = TP / (TP + FP)
        #Specificity = TN / (TN + FP)
        # F1 score = Dice
        Dice = 2 * Precision * Recall / (Precision + Recall)
        # Overall accuracy
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        # IoU for poly
        IoU_polyp = TP / (TP + FP + FN)

        return MAE.data.cpu().numpy().squeeze(), Recall.data.cpu().numpy().squeeze(), Precision.data.cpu().numpy().squeeze(), Accuracy.data.cpu().numpy().squeeze(), Dice.data.cpu().numpy().squeeze(), IoU_polyp.data.cpu().numpy().squeeze()

        
    def update(self, pred, gt):
        mae, recall, precision, accuracy, dice, ioU_polyp = self.evaluate(pred, gt)        
        self.MAE.append(mae)
        self.Recall.append(recall)
        self.Precision.append(precision)
        self.Accuracy.append(accuracy)
        self.Dice.append(dice)       
        self.IoU_polyp.append(ioU_polyp)

    def show(self):
        print("MAE: " + str(round(np.mean(self.MAE)*100,2)),
      "  Recall: " + str(round(np.mean(self.Recall)*100,2)), 
      "  Pre: " + str(round(np.mean(self.Precision)*100,2)),
      "  Acc: " + str(round(np.mean(self.Accuracy)*100,2)),
      "  Dice: " + str(round(np.mean(self.Dice)*100,2)),
      "  IoU: " + str(round(np.mean(self.IoU_polyp)*100,2)))
        print('\n')

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def sigmoid_rampup(current, rampup_length):
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(consistency,epoch,consistency_rampup):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def compute_sdf(img_gt, out_shape):
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): 
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
    return normalized_sdf
    
def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = torch.mean ((input_softmax-target_softmax)**2)
    return mse_loss

 
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
'''
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
'''        
        
def save_results(images,probs,lables,save_dir,h,w,i):
    #print(probs.shape)
    pred = np.argmax(probs,axis=0)
    pred_vis = np.zeros((h,w,3),np.uint8)
    pred_vis[pred==1]=[255,0,0]
    pred_vis[pred==2]=[0,255,0]
    pred_vis[pred==3]=[0,0,255]
    pred_vis[pred==4]=[255,0,255]
    cv2.imwrite(save_dir+'Pred'+str(i)+'.png',pred_vis[:,:,::-1])