import torch
import torch.nn as nn
import numpy as np


class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, gt, mask):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum, dict(l1_loss=mask_sum)
        else:
        device = pred.device
        dtype = pred.dtype
        if not isinstance(gt, torch.Tensor):
            if isinstance(gt, np.ndarray):
                gt = torch.from_numpy(gt).to(device=device, dtype=dtype)
            else:
                gt = torch.as_tensor(gt, device=device, dtype=dtype)
        else:
            gt = gt.to(device=device, dtype=dtype)

        if not isinstance(mask, torch.Tensor):
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(device=device, dtype=dtype)
            else:
                mask = torch.as_tensor(mask, device=device, dtype=dtype)
        else:
            mask = mask.to(device=device, dtype=dtype)

        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask[:, 0]

        loss = (torch.abs(pred[:, 0] - gt) * mask).sum() / mask_sum
            return loss, dict(l1_loss=loss)


class BalanceL1Loss(nn.Module):
    def __init__(self, negative_ratio=3.):
        super(BalanceL1Loss, self).__init__()
        self.negative_ratio = negative_ratio

    def forward(self, pred: torch.Tensor, gt, mask):
        '''
        Args:
            pred: (N, 1, H, W).
            gt: (N, H, W).
            mask: (N, H, W).
        '''
        loss = torch.abs(pred[:, 0] - gt)
        positive = loss * mask
        negative = loss * (1 - mask)
        positive_count = int(mask.sum())
        negative_count = min(
                int((1 - mask).sum()),
                int(positive_count * self.negative_ratio))
        negative_loss, _ = torch.topk(negative.view(-1), negative_count)
        negative_loss = negative_loss.sum() / negative_count
        positive_loss = positive.sum() / positive_count
        return positive_loss + negative_loss,\
            dict(l1_loss=positive_loss, nge_l1_loss=negative_loss)
