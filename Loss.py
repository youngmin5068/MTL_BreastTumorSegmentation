import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)
        dice = (2 * intersection + 1e-8) / (union + 1e-8)  # Adding a small epsilon to avoid division by zero
        dice_loss = 1 - dice
        return dice_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        TP = torch.sum(inputs * targets)
        FP = torch.sum(inputs * (1 - targets))
        FN = torch.sum((1 - inputs) * targets)

        tversky_index = (TP) / (TP + self.alpha * FN + self.beta * FP + 1e-8)
        tversky_loss = 1 - tversky_index

        return tversky_loss
