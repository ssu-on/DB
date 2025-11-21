import sys

import torch
import torch.nn as nn
import numpy as np

# Import SubtitleRefinedL1BalanceCELoss from subtitle_refined_loss module
from .subtitle_refined_loss import SubtitleRefinedL1BalanceCELoss
from .subtitle_color_loss import SubtitleColorConsistencyLoss


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

# @@ for subtitle branch loss
class SubtitleBranchLoss(nn.Module):
    '''
    Loss function for subtitle branch.
    
    Combines:
    1. Subtitle Binary Loss (BCE/Dice) - direct supervision for subtitle binary map
    2. Color Embedding Loss - self-refinement for subtitle feature clustering
    
    Loss = L_subtitle_binary + λ * L_color_embedding
    
    This loss is designed for Stage 2 training where:
    - backbone/fuse are frozen
    - Only subtitle_fuse_branch + subtitle_binary_head + subtitle_color_embed_head are trained
    '''
    
    def __init__(self, 
                 binary_loss_type='bce',  # 'bce', 'dice', or 'bce+dice'
                 lambda_color=0.3,
                 eps=1e-6,
                 **kwargs):
        super(SubtitleBranchLoss, self).__init__()
        
        from .dice_loss import DiceLoss
        from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
        from .subtitle_color_loss import SubtitleColorConsistencyLoss
        
        self.binary_loss_type = binary_loss_type
        self.lambda_color = lambda_color
        
        # Binary loss components
        if binary_loss_type == 'bce':
            self.binary_loss = BalanceCrossEntropyLoss(**kwargs.get('bce_kwargs', {}))
            self.use_dice = False
        elif binary_loss_type == 'dice':
            self.binary_loss = DiceLoss(eps=eps)
            self.use_dice = True
        elif binary_loss_type == 'bce+dice':
            self.bce_loss = BalanceCrossEntropyLoss(**kwargs.get('bce_kwargs', {}))
            self.dice_loss = DiceLoss(eps=eps)
            self.use_dice = True
        else:
            raise ValueError(f"binary_loss_type must be 'bce', 'dice', or 'bce+dice', got {binary_loss_type}")
        
        # Color embedding loss
        self.color_loss = SubtitleColorConsistencyLoss(**kwargs.get('color_kwargs', {}))
    
    def _to_tensor(self, value, device, dtype=torch.float32):
        """Convert various input types to torch.Tensor."""
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=device, dtype=dtype)
        if isinstance(value, (list, tuple)):
            tensors = [self._to_tensor(v, device, dtype) for v in value]
            return torch.stack(tensors, dim=0)
        return torch.as_tensor(value, device=device, dtype=dtype)
    
    def forward(self, pred, batch):
        """
        Args:
            pred: dict containing
                - subtitle_binary: (N, 1, H, W) subtitle binary prediction
                - subtitle_color_embedding: (N, C, H, W) subtitle color embedding
            batch: dict containing
                - gt: (N, 1, H, W) ground truth subtitle mask
                - mask: (N, H, W) ignore mask
        Returns:
            (loss, metrics) tuple
        """
        assert 'subtitle_binary' in pred, "subtitle_binary must be in pred"
        assert 'subtitle_color_embedding' in pred, "subtitle_color_embedding must be in pred"
        
        subtitle_binary = pred['subtitle_binary']
        subtitle_color_embedding = pred['subtitle_color_embedding']
        
        # Convert batch values to tensors (handle list/numpy array inputs)
        device = subtitle_binary.device
        dtype = subtitle_binary.dtype
        gt = self._to_tensor(batch['gt'], device=device, dtype=dtype)
        mask = self._to_tensor(batch['mask'], device=device, dtype=dtype)
        
        # Ensure gt has shape (N, 1, H, W) and mask has shape (N, H, W)
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask[:, 0]  # (N, 1, H, W) -> (N, H, W)
        
        # Compute binary loss
        if self.binary_loss_type == 'bce':
            binary_loss = self.binary_loss(subtitle_binary, gt, mask)
        elif self.binary_loss_type == 'dice':
            binary_loss = self.binary_loss(subtitle_binary, gt, mask)
        elif self.binary_loss_type == 'bce+dice':
            bce_loss = self.bce_loss(subtitle_binary, gt, mask)
            dice_loss = self.dice_loss(subtitle_binary, gt, mask)
            binary_loss = bce_loss + dice_loss
        else:
            raise ValueError(f"Invalid binary_loss_type: {self.binary_loss_type}")
        
        # Compute color embedding loss
        color_pred = {'color_embedding': subtitle_color_embedding}
        color_loss, color_metrics = self.color_loss(color_pred, batch)
        
        # Combine losses
        total_loss = binary_loss + self.lambda_color * color_loss
        
        # Build metrics
        metrics = {
            'subtitle_binary_loss': binary_loss.detach(),
            'subtitle_color_loss': color_loss.detach(),
            'subtitle_total_loss': total_loss.detach()
        }
        metrics.update({f'color_{k}': v for k, v in color_metrics.items()})
        
        if self.binary_loss_type == 'bce+dice':
            metrics['subtitle_bce_loss'] = bce_loss.detach()
            metrics['subtitle_dice_loss'] = dice_loss.detach()
        
        return total_loss, metrics
