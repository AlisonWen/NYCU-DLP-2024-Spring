import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred, target, epsilon=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + epsilon) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + epsilon)))
    
    return loss.mean()

def calculate_accuracy(pred, target):
    pred = torch.sigmoid(pred)  # Apply sigmoid to get [0,1] range
    preds_binary = (pred > 0.5).float()  # Threshold predictions
    correct = (preds_binary == target).float()  # Correct predictions
    accuracy = correct.sum() / (target.size(0) * target.size(2) * target.size(3))
    return accuracy

def dice_score(pred, target):
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    score = (2. * intersection) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2))
    return score