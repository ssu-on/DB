# import sys

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class SegDetectorLossBuilder():
#     '''
#     Build loss functions for SegDetector.
#     Details about the built functions:
#         Input:
#             pred: A dict which contains predictions.
#                 thresh: The threshold prediction
#                 binary: The text segmentation prediction.
#                 thresh_binary: Value produced by `step_function(binary - thresh)`.
#             batch:
#                 gt: Text regions bitmap gt.
#                 mask: Ignore mask,
#                     pexels where value is 1 indicates no contribution to loss.
#                 thresh_mask: Mask indicates regions cared by thresh supervision.
#                 thresh_map: Threshold gt.
#         Return:
#             (loss, metrics).
#             loss: A scalar loss value.
#             metrics: A dict contraining partial loss values.
#     '''

#     def __init__(self, loss_class, *args, **kwargs):
#         self.loss_class = loss_class
#         self.loss_args = args
#         self.loss_kwargs = kwargs

#     def build(self):
#         return getattr(sys.modules[__name__], self.loss_class)(*self.loss_args, **self.loss_kwargs)


# class DiceLoss(nn.Module):
#     '''
#     DiceLoss on binary.
#     For SegDetector without adaptive module.
#     '''

#     def __init__(self, eps=1e-6):
#         super(DiceLoss, self).__init__()
#         from .dice_loss import DiceLoss as Loss
#         self.loss = Loss(eps)

#     def forward(self, pred, batch):
#         loss = self.loss(pred['binary'], batch['gt'], batch['mask'])
#         return loss, dict(dice_loss=loss)


# class BalanceBCELoss(nn.Module):
#     '''
#     DiceLoss on binary.
#     For SegDetector without adaptive module.
#     '''

#     def __init__(self, eps=1e-6):
#         super(BalanceBCELoss, self).__init__()
#         from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
#         self.loss = BalanceCrossEntropyLoss()

#     def forward(self, pred, batch):
#         loss = self.loss(pred['binary'], batch['gt'], batch['mask'])
#         return loss, dict(dice_loss=loss)


# class AdaptiveDiceLoss(nn.Module):
#     '''
#     Integration of DiceLoss on both binary
#         prediction and thresh prediction.
#     '''

#     def __init__(self, eps=1e-6):
#         super(AdaptiveDiceLoss, self).__init__()
#         from .dice_loss import DiceLoss
#         self.main_loss = DiceLoss(eps)
#         self.thresh_loss = DiceLoss(eps)

#     def forward(self, pred, batch):
#         assert isinstance(pred, dict)
#         assert 'binary' in pred
#         assert 'thresh_binary' in pred

#         binary = pred['binary']
#         thresh_binary = pred['thresh_binary']
#         gt = batch['gt']
#         mask = batch['mask']
#         main_loss = self.main_loss(binary, gt, mask)
#         thresh_loss = self.thresh_loss(thresh_binary, gt, mask)
#         loss = main_loss + thresh_loss
#         return loss, dict(main_loss=main_loss, thresh_loss=thresh_loss)


# class AdaptiveInstanceDiceLoss(nn.Module):
#     '''
#     InstanceDiceLoss on both binary and thresh_bianry.
#     '''

#     def __init__(self, iou_thresh=0.2, thresh=0.3):
#         super(AdaptiveInstanceDiceLoss, self).__init__()
#         from .dice_loss import InstanceDiceLoss, DiceLoss
#         self.main_loss = DiceLoss()
#         self.main_instance_loss = InstanceDiceLoss()
#         self.thresh_loss = DiceLoss()
#         self.thresh_instance_loss = InstanceDiceLoss()
#         self.weights = nn.ParameterDict(dict(
#             main=nn.Parameter(torch.ones(1)),
#             thresh=nn.Parameter(torch.ones(1)),
#             main_instance=nn.Parameter(torch.ones(1)),
#             thresh_instance=nn.Parameter(torch.ones(1))))

#     def partial_loss(self, weight, loss):
#         return loss / weight + torch.log(torch.sqrt(weight))

#     def forward(self, pred, batch):
#         main_loss = self.main_loss(pred['binary'], batch['gt'], batch['mask'])
#         thresh_loss = self.thresh_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
#         main_instance_loss = self.main_instance_loss(
#             pred['binary'], batch['gt'], batch['mask'])
#         thresh_instance_loss = self.thresh_instance_loss(
#             pred['thresh_binary'], batch['gt'], batch['mask'])
#         loss = self.partial_loss(self.weights['main'], main_loss) \
#                + self.partial_loss(self.weights['thresh'], thresh_loss) \
#                + self.partial_loss(self.weights['main_instance'], main_instance_loss) \
#                + self.partial_loss(self.weights['thresh_instance'], thresh_instance_loss)
#         metrics = dict(
#             main_loss=main_loss,
#             thresh_loss=thresh_loss,
#             main_instance_loss=main_instance_loss,
#             thresh_instance_loss=thresh_instance_loss)
#         metrics.update(self.weights)
#         return loss, metrics


# class L1DiceLoss(nn.Module):
#     '''
#     L1Loss on thresh, DiceLoss on thresh_binary and binary.
#     '''

#     def __init__(self, eps=1e-6, l1_scale=10):
#         super(L1DiceLoss, self).__init__()
#         self.dice_loss = AdaptiveDiceLoss(eps=eps)
#         from .l1_loss import MaskL1Loss
#         self.l1_loss = MaskL1Loss()
#         self.l1_scale = l1_scale

#     def forward(self, pred, batch):
#         dice_loss, metrics = self.dice_loss(pred, batch)
#         l1_loss, l1_metric = self.l1_loss(
#             pred['thresh'], batch['thresh_map'], batch['thresh_mask'])

#         loss = dice_loss + self.l1_scale * l1_loss
#         metrics.update(**l1_metric)
#         return loss, metrics


# class FullL1DiceLoss(L1DiceLoss):
#     '''
#     L1loss on thresh, pixels with topk losses in non-text regions are also counted.
#     DiceLoss on thresh_binary and binary.
#     '''

#     def __init__(self, eps=1e-6, l1_scale=10):
#         nn.Module.__init__(self)
#         self.dice_loss = AdaptiveDiceLoss(eps=eps)
#         from .l1_loss import BalanceL1Loss
#         self.l1_loss = BalanceL1Loss()
#         self.l1_scale = l1_scale


# class L1BalanceCELoss(nn.Module):
#     '''
#     Balanced CrossEntropy Loss on `binary`,
#     MaskL1Loss on `thresh`,
#     DiceLoss on `thresh_binary`.
#     Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
#     '''

#     def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
#         super(L1BalanceCELoss, self).__init__()
#         from .dice_loss import DiceLoss
#         from .l1_loss import MaskL1Loss
#         from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
#         self.dice_loss = DiceLoss(eps=eps)
#         self.l1_loss = MaskL1Loss()
#         self.bce_loss = BalanceCrossEntropyLoss()

#         self.l1_scale = l1_scale
#         self.bce_scale = bce_scale

#     def forward(self, pred, batch):
#         bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
#         metrics = dict(bce_loss=bce_loss)
#         if 'thresh' in pred:
#             l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
#             dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
#             metrics['thresh_loss'] = dice_loss
#             loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
#             metrics.update(**l1_metric)
#         else:
#             loss = bce_loss
#         return loss, metrics


# class L1BCEMiningLoss(nn.Module):
#     '''
#     Basicly the same with L1BalanceCELoss, where the bce loss map is used as
#         attention weigts for DiceLoss
#     '''

#     def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
#         super(L1BCEMiningLoss, self).__init__()
#         from .dice_loss import DiceLoss
#         from .l1_loss import MaskL1Loss
#         from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
#         self.dice_loss = DiceLoss(eps=eps)
#         self.l1_loss = MaskL1Loss()
#         self.bce_loss = BalanceCrossEntropyLoss()

#         self.l1_scale = l1_scale
#         self.bce_scale = bce_scale

#     def forward(self, pred, batch):
#         bce_loss, bce_map = self.bce_loss(pred['binary'], batch['gt'], batch['mask'],
#                                           return_origin=True)
#         l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
#         bce_map = (bce_map - bce_map.min()) / (bce_map.max() - bce_map.min())
#         dice_loss = self.dice_loss(
#             pred['thresh_binary'], batch['gt'],
#             batch['mask'], weights=bce_map + 1)
#         metrics = dict(bce_loss=bce_loss)
#         metrics['thresh_loss'] = dice_loss
#         loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
#         metrics.update(**l1_metric)
#         return loss, metrics


# class L1LeakyDiceLoss(nn.Module):
#     '''
#     LeakyDiceLoss on binary,
#     MaskL1Loss on thresh,
#     DiceLoss on thresh_binary.
#     '''

#     def __init__(self, eps=1e-6, coverage_scale=5, l1_scale=10):
#         super(L1LeakyDiceLoss, self).__init__()
#         from .dice_loss import DiceLoss, LeakyDiceLoss
#         from .l1_loss import MaskL1Loss
#         self.main_loss = LeakyDiceLoss(coverage_scale=coverage_scale)
#         self.l1_loss = MaskL1Loss()
#         self.thresh_loss = DiceLoss(eps=eps)

#         self.l1_scale = l1_scale

#     def forward(self, pred, batch):
#         main_loss, metrics = self.main_loss(pred['binary'], batch['gt'], batch['mask'])
#         thresh_loss = self.thresh_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
#         l1_loss, l1_metric = self.l1_loss(
#             pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
#         metrics.update(**l1_metric, thresh_loss=thresh_loss)
#         loss = main_loss + thresh_loss + l1_loss * self.l1_scale
#         return loss, metrics


# class SubtitleBoundaryAwareL1BalanceCELoss(nn.Module):
#     """
#     DBNet 기본 손실(L1BalanceCELoss)에 
#     subtitle 경계 영역(G - Gs)에 대한 soft boundary supervision을 추가.
    
#     pred['binary']는 항상 sigmoid 적용된 확률 맵임.
#     """

#     def __init__(
#         self,
#         eps=1e-6,
#         l1_scale=10,
#         bce_scale=5,
#         boundary_scale=0.2,
#         boundary_target_scale=0.3,
#     ):
#         super().__init__()
#         self.base_loss = L1BalanceCELoss(
#             eps=eps,
#             l1_scale=l1_scale,
#             bce_scale=bce_scale
#         )
#         self.boundary_scale = boundary_scale
#         self.boundary_target_scale = boundary_target_scale
#         self.eps = eps

#     def forward(self, pred, batch):
#         base_loss, metrics = self.base_loss(pred, batch)
#         metrics = dict(metrics)

#         binary_pred = pred.get("binary", None)
#         boundary = batch.get("boundary", None)

#         def resolve_device():
#             if binary_pred is not None and torch.is_tensor(binary_pred):
#                 return binary_pred.device
#             for value in pred.values():
#                 if torch.is_tensor(value):
#                     return value.device
#             return torch.device("cpu")

#         device = resolve_device()
#         zero = torch.zeros((), device=device)

#         if binary_pred is None or boundary is None or self.boundary_scale <= 0:
#             metrics["boundary_soft_loss"] = zero
#             return base_loss, metrics

#         # ---------- Shape 정리 ----------
#         if binary_pred.dim() == 3:
#             binary_pred = binary_pred.unsqueeze(1)
#         if boundary.dim() == 3:
#             boundary = boundary.unsqueeze(1)

#         spatial = binary_pred.shape[2:]
#         # boundary는 geometry 유지 → nearest
#         boundary = F.interpolate(boundary.float(), size=spatial, mode="nearest")
#         boundary = boundary.to(device=device, dtype=binary_pred.dtype)
#         binary_pred = binary_pred.to(device=device, dtype=binary_pred.dtype)

#         # ---------- Soft boundary weight ----------
#         boundary_w = boundary.clamp(0., 1.)

#         # soft target 생성 (0 ~ boundary_target_scale)
#         target = boundary_w * self.boundary_target_scale

#         # ---------- BCE (sigmoid output이므로 normal BCE) ----------
#         bce_map = F.binary_cross_entropy(binary_pred, target, reduction="none")

#         # ---------- Per-sample normalization ----------
#         bw_sum = boundary_w.sum((1,2,3))
#         valid = bw_sum > self.eps

#         if not torch.any(valid):
#             metrics["boundary_soft_loss"] = zero
#             return base_loss, metrics

#         loss_per_sample = (bce_map * boundary_w).sum((1,2,3)) / (bw_sum + self.eps)
#         boundary_loss = loss_per_sample[valid].mean()

#         total_loss = base_loss + self.boundary_scale * boundary_loss

#         metrics["boundary_soft_loss"] = boundary_loss

#         return total_loss, metrics
import sys

import torch
import torch.nn as nn


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
