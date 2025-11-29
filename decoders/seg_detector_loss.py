import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
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
    Subtitle-only loss for the subtitle style head.

    L = L_bce(S, gt_subtitle) + lambda_style * L_intra(E)

    - S: subtitle likelihood map ('subtitle_s'), resized to match gt resolution.
    - E: pixel-level style embedding ('subtitle_embedding'), defined on top of F5.
    - gt_subtitle: subtitle region mask (batch['gt']); for Stage 2 this is subtitle-only.
    """

    def __init__(self, bce_scale: float = 1.0, style_scale: float = 0.1, eps: float = 1e-6):
        super().__init__()
        from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
        self.bce_loss_fn = BalanceCrossEntropyLoss()
        self.bce_scale = bce_scale
        self.style_scale = style_scale
        self.eps = eps

    def _to_tensor(self, value, device, dtype=torch.float32):
        """
        Robustly convert list / numpy / tensor to a batched tensor on the right device.
        This matches the behavior used in SubtitleRefinedL1BalanceCELoss.
        """
        import numpy as np

        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=device, dtype=dtype)
        if isinstance(value, (list, tuple)):
            tensors = [self._to_tensor(v, device, dtype) for v in value]
            return torch.stack(tensors, dim=0)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _style_variance_loss(self, embedding: torch.Tensor, subtitle_mask: torch.Tensor):
        """
        Per-frame subtitle style variance:
        for each image, compute the variance of pixel embeddings inside subtitle regions.
        """
        N, C, H, W = embedding.shape
        # subtitle_mask: (N, 1, H, W) or (N, H, W)
        if subtitle_mask.dim() == 4:
            subtitle_mask_ = subtitle_mask.squeeze(1)
        else:
            subtitle_mask_ = subtitle_mask

        total_var = embedding.new_zeros(())
        count = 0

        for b in range(N):
            mask = (subtitle_mask_[b] > 0.5)
            num_pixels = mask.sum().item()
            if num_pixels < 1:
                continue

            feat_b = embedding[b]  # (C, H, W)
            feat_flat = feat_b.view(C, -1)
            mask_flat = mask.view(-1)
            feat_sub = feat_flat[:, mask_flat]  # (C, M)
            if feat_sub.numel() == 0:
                continue
            mean = feat_sub.mean(dim=1, keepdim=True)  # (C, 1)
            var = ((feat_sub - mean) ** 2).sum() / (feat_sub.shape[1] + self.eps)
            total_var = total_var + var
            count += 1

        if count == 0:
            return embedding.new_zeros(())
        return total_var / count

    def forward(self, pred, batch):
        assert isinstance(pred, dict), "SubtitleBranchLoss expects dict prediction"
        if "subtitle_s" not in pred or "subtitle_embedding" not in pred:
            raise KeyError("SubtitleBranchLoss requires 'subtitle_s' and 'subtitle_embedding' in pred")

        s_map = pred["subtitle_s"]              # (N, 1, Hs, Ws)
        e_map = pred["subtitle_embedding"]      # (N, D, He, We)

        # batch['gt'], batch['mask'] may be list / numpy / tensor depending on loader.
        # Convert them robustly to tensors on the same device as s_map.
        device = s_map.device
        dtype = s_map.dtype
        gt = self._to_tensor(batch["gt"], device=device, dtype=dtype)   # (N, 1, Hg, Wg)
        mask_raw = batch.get("mask", None)

        # Resize gt/mask to match S resolution for BCE term (use nearest to preserve binary nature)
        target_size_s = s_map.shape[-2:]
        gt_small = F.interpolate(gt, size=target_size_s, mode="nearest")
        if mask_raw is not None:
            mask = self._to_tensor(mask_raw, device=device, dtype=dtype)
            # BalanceCrossEntropyLoss expects mask of shape (N, H, W)
            # so we always squeeze the channel dimension after interpolation.
            if mask.dim() == 3:  # (N, H, W)
                mask_small = F.interpolate(
                    mask.unsqueeze(1), size=target_size_s, mode="nearest"
                ).squeeze(1)
            else:  # (N, 1, H, W) or similar
                mask_small = F.interpolate(
                    mask, size=target_size_s, mode="nearest"
                ).squeeze(1)
        else:
            # Default: use all pixels (mask = 1) with shape (N, H, W)
            mask_small = torch.ones_like(gt_small[:, 0, :, :])

        # ---- Debug / monitoring stats for subtitle branch ----
        with torch.no_grad():
            # Positive subtitle pixels in GT at S-map resolution
            gt_pos = (gt_small > 0.5).float().sum()
            gt_total = float(gt_small.numel())
            gt_pos_ratio = gt_pos / (gt_total + self.eps)

            # Valid pixels according to mask
            mask_pos = mask_small.float().sum()
            mask_total = float(mask_small.numel())
            mask_pos_ratio = mask_pos / (mask_total + self.eps)

            s_min = s_map.min()
            s_max = s_map.max()
            s_mean = s_map.mean()

        # 1) BCE loss on subtitle likelihood map
        bce_loss = self.bce_loss_fn(s_map, gt_small, mask_small)

        # 2) Style consistency loss (variance inside subtitle regions)
        #    Use gt resized to the embedding resolution, which may differ from s_map.
        target_size_e = e_map.shape[-2:]
        gt_for_style = F.interpolate(gt, size=target_size_e, mode="nearest")
        style_loss = self._style_variance_loss(e_map, gt_for_style)

        loss = self.bce_scale * bce_loss + self.style_scale * style_loss

        metrics = dict(
            subtitle_loss=loss.detach(),
            subtitle_bce=bce_loss.detach(),
            subtitle_style=style_loss.detach(),
            # Raw S-map statistics
            subtitle_s_mean=s_mean.detach(),
            subtitle_s_min=s_min.detach(),
            subtitle_s_max=s_max.detach(),
            # GT / mask coverage at S resolution
            subtitle_gt_pos_ratio=gt_pos_ratio.detach(),
            subtitle_mask_pos_ratio=mask_pos_ratio.detach(),
        )
        # Optional: monitor how many pixels survive in subtitle_binary
        if "subtitle_binary" in pred:
            subtitle_binary = pred["subtitle_binary"]
            metrics["subtitle_bin_mean"] = subtitle_binary.mean().detach()
        return loss, metrics
