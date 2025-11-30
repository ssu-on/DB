import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SegDetectorLossBuilder():
    '''
    Build loss functions for SegDetector.
    Details about the built functions:
        Input:
            pred: A dict which contains predictions.
                thresh: The threshold prediction
                binary: The text segmentation prediction.
                thresh_binary: Value produced by `step_function(binary - thresh)`.
            batch:
                gt: Text regions bitmap gt.
                mask: Ignore mask,
                    pexels where value is 1 indicates no contribution to loss.
                thresh_mask: Mask indicates regions cared by thresh supervision.
                thresh_map: Threshold gt.
        Return:
            (loss, metrics).
            loss: A scalar loss value.
            metrics: A dict contraining partial loss values.
    '''

    def __init__(self, loss_class, *args, **kwargs):
        self.loss_class = loss_class
        self.loss_args = args
        self.loss_kwargs = kwargs

    def build(self):
        return getattr(sys.modules[__name__], self.loss_class)(*self.loss_args, **self.loss_kwargs)


class DiceLoss(nn.Module):
    '''
    DiceLoss on binary.
    For SegDetector without adaptive module.
    '''

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        from .dice_loss import DiceLoss as Loss
        self.loss = Loss(eps)

    def forward(self, pred, batch):
        loss = self.loss(pred['binary'], batch['gt'], batch['mask'])
        return loss, dict(dice_loss=loss)


class BalanceBCELoss(nn.Module):
    '''
    DiceLoss on binary.
    For SegDetector without adaptive module.
    '''

    def __init__(self, eps=1e-6):
        super(BalanceBCELoss, self).__init__()
        from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
        self.loss = BalanceCrossEntropyLoss()

    def forward(self, pred, batch):
        loss = self.loss(pred['binary'], batch['gt'], batch['mask'])
        return loss, dict(dice_loss=loss)


class AdaptiveDiceLoss(nn.Module):
    '''
    Integration of DiceLoss on both binary
        prediction and thresh prediction.
    '''

    def __init__(self, eps=1e-6):
        super(AdaptiveDiceLoss, self).__init__()
        from .dice_loss import DiceLoss
        self.main_loss = DiceLoss(eps)
        self.thresh_loss = DiceLoss(eps)

    def forward(self, pred, batch):
        assert isinstance(pred, dict)
        assert 'binary' in pred
        assert 'thresh_binary' in pred

        binary = pred['binary']
        thresh_binary = pred['thresh_binary']
        gt = batch['gt']
        mask = batch['mask']
        main_loss = self.main_loss(binary, gt, mask)
        thresh_loss = self.thresh_loss(thresh_binary, gt, mask)
        loss = main_loss + thresh_loss
        return loss, dict(main_loss=main_loss, thresh_loss=thresh_loss)


class AdaptiveInstanceDiceLoss(nn.Module):
    '''
    InstanceDiceLoss on both binary and thresh_bianry.
    '''

    def __init__(self, iou_thresh=0.2, thresh=0.3):
        super(AdaptiveInstanceDiceLoss, self).__init__()
        from .dice_loss import InstanceDiceLoss, DiceLoss
        self.main_loss = DiceLoss()
        self.main_instance_loss = InstanceDiceLoss()
        self.thresh_loss = DiceLoss()
        self.thresh_instance_loss = InstanceDiceLoss()
        self.weights = nn.ParameterDict(dict(
            main=nn.Parameter(torch.ones(1)),
            thresh=nn.Parameter(torch.ones(1)),
            main_instance=nn.Parameter(torch.ones(1)),
            thresh_instance=nn.Parameter(torch.ones(1))))

    def partial_loss(self, weight, loss):
        return loss / weight + torch.log(torch.sqrt(weight))

    def forward(self, pred, batch):
        main_loss = self.main_loss(pred['binary'], batch['gt'], batch['mask'])
        thresh_loss = self.thresh_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
        main_instance_loss = self.main_instance_loss(
            pred['binary'], batch['gt'], batch['mask'])
        thresh_instance_loss = self.thresh_instance_loss(
            pred['thresh_binary'], batch['gt'], batch['mask'])
        loss = self.partial_loss(self.weights['main'], main_loss) \
               + self.partial_loss(self.weights['thresh'], thresh_loss) \
               + self.partial_loss(self.weights['main_instance'], main_instance_loss) \
               + self.partial_loss(self.weights['thresh_instance'], thresh_instance_loss)
        metrics = dict(
            main_loss=main_loss,
            thresh_loss=thresh_loss,
            main_instance_loss=main_instance_loss,
            thresh_instance_loss=thresh_instance_loss)
        metrics.update(self.weights)
        return loss, metrics


class L1DiceLoss(nn.Module):
    '''
    L1Loss on thresh, DiceLoss on thresh_binary and binary.
    '''

    def __init__(self, eps=1e-6, l1_scale=10):
        super(L1DiceLoss, self).__init__()
        self.dice_loss = AdaptiveDiceLoss(eps=eps)
        from .l1_loss import MaskL1Loss
        self.l1_loss = MaskL1Loss()
        self.l1_scale = l1_scale

    def forward(self, pred, batch):
        dice_loss, metrics = self.dice_loss(pred, batch)
        l1_loss, l1_metric = self.l1_loss(
            pred['thresh'], batch['thresh_map'], batch['thresh_mask'])

        loss = dice_loss + self.l1_scale * l1_loss
        metrics.update(**l1_metric)
        return loss, metrics


class FullL1DiceLoss(L1DiceLoss):
    '''
    L1loss on thresh, pixels with topk losses in non-text regions are also counted.
    DiceLoss on thresh_binary and binary.
    '''

    def __init__(self, eps=1e-6, l1_scale=10):
        nn.Module.__init__(self)
        self.dice_loss = AdaptiveDiceLoss(eps=eps)
        from .l1_loss import BalanceL1Loss
        self.l1_loss = BalanceL1Loss()
        self.l1_scale = l1_scale


class L1BalanceCELoss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    #def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5, **kwargs):
        super(L1BalanceCELoss, self).__init__()
        from .dice_loss import DiceLoss
        from .l1_loss import MaskL1Loss
        from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, batch):
        bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
        metrics = dict(bce_loss=bce_loss)
        if 'thresh' in pred:
            l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
            dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
            metrics['thresh_loss'] = dice_loss
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
            metrics.update(**l1_metric)
        else:
            loss = bce_loss
        return loss, metrics


class L1BCEMiningLoss(nn.Module):
    '''
    Basicly the same with L1BalanceCELoss, where the bce loss map is used as
        attention weigts for DiceLoss
    '''

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(L1BCEMiningLoss, self).__init__()
        from .dice_loss import DiceLoss
        from .l1_loss import MaskL1Loss
        from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, batch):
        bce_loss, bce_map = self.bce_loss(pred['binary'], batch['gt'], batch['mask'],
                                          return_origin=True)
        l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
        bce_map = (bce_map - bce_map.min()) / (bce_map.max() - bce_map.min())
        dice_loss = self.dice_loss(
            pred['thresh_binary'], batch['gt'],
            batch['mask'], weights=bce_map + 1)
        metrics = dict(bce_loss=bce_loss)
        metrics['thresh_loss'] = dice_loss
        loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        metrics.update(**l1_metric)
        return loss, metrics


class L1LeakyDiceLoss(nn.Module):
    '''
    LeakyDiceLoss on binary,
    MaskL1Loss on thresh,
    DiceLoss on thresh_binary.
    '''
    
    def __init__(self, eps=1e-6, coverage_scale=5, l1_scale=10):
        super(L1LeakyDiceLoss, self).__init__()
        from .dice_loss import DiceLoss, LeakyDiceLoss
        from .l1_loss import MaskL1Loss
        self.main_loss = LeakyDiceLoss(coverage_scale=coverage_scale)
        self.l1_loss = MaskL1Loss()
        self.thresh_loss = DiceLoss(eps=eps)

        self.l1_scale = l1_scale

    def forward(self, pred, batch):
        main_loss, metrics = self.main_loss(pred['binary'], batch['gt'], batch['mask'])
        thresh_loss = self.thresh_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
        l1_loss, l1_metric = self.l1_loss(
            pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
        metrics.update(**l1_metric, thresh_loss=thresh_loss)
        loss = main_loss + thresh_loss + l1_loss * self.l1_scale
        return loss, metrics

class SubtitleBranchLoss(nn.Module):
    """
    Minimal subtitle-only loss:
        L = L_sub_detect + λ_intra * L_intra + λ_inter * L_inter
    """

    def __init__(self,
                 bce_weight=1.0,
                 dice_weight=1.0,
                 lambda_intra=1.0,
                 lambda_inter=1.0,
                 margin=0.5,
                 text_binary_threshold=0.3,
                 eps=1e-6):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.lambda_intra = float(lambda_intra)
        self.lambda_inter = float(lambda_inter)
        self.margin = float(margin)
        self.text_binary_threshold = float(text_binary_threshold)
        self.eps = float(eps)

    @staticmethod
    def _to_tensor(value, device, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=device, dtype=dtype)
        if isinstance(value, (list, tuple)):
            tensors = [SubtitleBranchLoss._to_tensor(v, device, dtype) for v in value]
            return torch.stack(tensors, dim=0)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _dice_loss(self, pred, target, mask):
        pred = pred * mask
        target = target * mask
        intersection = (pred * target).sum()
        denom = pred.sum() + target.sum() + self.eps
        dice = 1 - (2 * intersection + self.eps) / denom
        return dice

    def _masked_mean(self, feat, weight):
        """feat: (N,C,H,W), weight: (N,1,H,W)."""
        weight_sum = weight.sum(dim=(2, 3)).clamp_min(self.eps)
        mean = (feat * weight).sum(dim=(2, 3)) / weight_sum
        return mean, weight_sum

    def _intra_loss(self, feat, weight):
        # weight: (N,1,H,W)
        if weight.sum() < self.eps:
            return feat.new_tensor(0.)
        mean, _ = self._masked_mean(feat, weight)
        diff = (feat - mean[:, :, None, None]) ** 2
        loss = (diff * weight).sum() / (weight.sum() * feat.size(1) + self.eps)
        return loss

    def _inter_loss(self, subtitle_feat, subtitle_center, scene_weight, valid_mask):
        """
        Compute distance between each scene-text feature (in subtitle space) and the
        subtitle center, encouraging scene pixels to stay beyond the margin.
        """
        if valid_mask.sum() == 0:
            return subtitle_feat.new_zeros(())
        center = subtitle_center[:, :, None, None]  # (N, C, 1, 1)
        diff_sq = (subtitle_feat - center) ** 2
        diff_sq = diff_sq.sum(dim=1)  # (N, H, W)
        weight = scene_weight.squeeze(1)
        weight_sum = weight.view(weight.size(0), -1).sum(dim=1).clamp_min(self.eps)
        dist = torch.sqrt(torch.clamp((diff_sq * weight).view(weight.size(0), -1).sum(dim=1) / weight_sum, min=0.0))
        valid_dist = dist[valid_mask]
        if valid_dist.numel() == 0:
            return dist.new_zeros(())
        loss = F.relu(self.margin - valid_dist).mean()
        return loss

    def forward(self, pred, batch):
        assert 'subtitle_binary' in pred, "subtitle_binary must be in pred"
        assert 'subtitle_feature' in pred, "subtitle_feature must be in pred"
        subtitle_binary = pred['subtitle_binary']
        subtitle_feature = pred['subtitle_feature']
        binary_pred = pred.get('binary', None)

        device = subtitle_binary.device
        dtype = subtitle_binary.dtype

        gt = self._to_tensor(batch['gt'], device=device, dtype=dtype)
        mask = self._to_tensor(batch['mask'], device=device, dtype=dtype)

        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 3 and mask.size(1) != 1:
            mask = mask.unsqueeze(1)
        mask = mask.float()

        subtitle_mask_full = (gt > 0.5).float()
        valid_mask_full = mask

        # Detection loss
        detect_loss = subtitle_binary.new_tensor(0.)
        metrics = {}
        if self.bce_weight > 0:
            bce = F.binary_cross_entropy(subtitle_binary, subtitle_mask_full, reduction='none')
            bce = (bce * valid_mask_full).sum() / (valid_mask_full.sum() + self.eps)
            detect_loss = detect_loss + self.bce_weight * bce
            metrics['subtitle_bce_loss'] = bce.detach()
        if self.dice_weight > 0:
            dice = self._dice_loss(subtitle_binary, subtitle_mask_full, valid_mask_full)
            detect_loss = detect_loss + self.dice_weight * dice
            metrics['subtitle_dice_loss'] = dice.detach()

        # Prepare downsampled masks for feature space
        feat_h, feat_w = subtitle_feature.shape[-2:]
        subtitle_mask_small = F.interpolate(subtitle_mask_full, size=(feat_h, feat_w), mode='nearest')
        valid_mask_small = F.interpolate(valid_mask_full, size=(feat_h, feat_w), mode='nearest')

        # Intra loss
        intra_weight = subtitle_mask_small * valid_mask_small
        L_intra = self._intra_loss(subtitle_feature, intra_weight)

        # Scene text mask using main binary prediction
        scene_weight = torch.zeros_like(intra_weight, device=device, dtype=dtype)
        if binary_pred is not None:
            if binary_pred.dim() == 4 and binary_pred.size(1) == 1:
                binary_pred_full = binary_pred.detach()
            elif binary_pred.dim() == 3:
                binary_pred_full = binary_pred.unsqueeze(1).detach()
            else:
                binary_pred_full = binary_pred[:, :1].detach()
            scene_mask_full = (binary_pred_full > self.text_binary_threshold).float() * (1.0 - subtitle_mask_full)
            scene_mask_full = scene_mask_full * valid_mask_full
            scene_weight = F.interpolate(scene_mask_full, size=(feat_h, feat_w), mode='nearest')

        # Inter loss
        L_inter = subtitle_feature.new_tensor(0.)
        valid_scene = (scene_weight.sum(dim=(2, 3)) > self.eps).squeeze(1)
        valid_sub = (intra_weight.sum(dim=(2, 3)) > self.eps).squeeze(1)
        valid_pairs = (valid_scene & valid_sub)
        if valid_pairs.any():
            sub_mean, _ = self._masked_mean(subtitle_feature, intra_weight)
            L_inter = self._inter_loss(subtitle_feature, sub_mean, scene_weight, valid_pairs)

        total_loss = detect_loss
        if self.lambda_intra > 0:
            total_loss = total_loss + self.lambda_intra * L_intra
        if self.lambda_inter > 0:
            total_loss = total_loss + self.lambda_inter * L_inter

        metrics.update({
            'subtitle_detect_loss': detect_loss.detach(),
            'subtitle_intra_loss': L_intra.detach(),
            'subtitle_inter_loss': L_inter.detach(),
            'subtitle_total_loss': total_loss.detach()
        })
        return total_loss, metrics
