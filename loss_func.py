import torch
import torch.nn.functional as F


def dice_loss(pred, target, eps=1e-6):
    B, C, H, W = pred.shape
    pred = F.softmax(pred, dim=1)

    ## target must be long for one_hot
    target = target.squeeze(1).long()

    target_onehot = F.one_hot(target, C).permute(0, 3, 1, 2).float()

    intersection = (pred * target_onehot).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()



def iou_loss(pred, target, eps=1e-6):
    """
    pred:   (B, C, H, W)
    target: (B, 1, H, W)
    """
    B, C, H, W = pred.shape
    pred = F.softmax(pred, dim=1)

    target = target.squeeze(1)
    target_onehot = F.one_hot(target, C).permute(0, 3, 1, 2).float()

    intersection = (pred * target_onehot).sum(dim=(0, 2, 3))
    union = (
        pred.sum(dim=(0, 2, 3))
        + target_onehot.sum(dim=(0, 2, 3))
        - intersection
    )

    iou = (intersection + eps) / (union + eps)
    return 1 - iou.mean()


def dice_iou_loss(pred, target):
    return dice_loss(pred, target) + iou_loss(pred, target)
