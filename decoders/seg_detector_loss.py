import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import SubtitleRefinedL1BalanceCELoss from subtitle_refined_loss module
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
                 negative_bce_alpha=0.5,  # Weight for scene text negative loss (critical for subtitle-only detection)
                 text_binary_threshold=0.5,  # Threshold for detecting all text regions from main binary
                 eps=1e-6,
                 **kwargs):
        super(SubtitleBranchLoss, self).__init__()
        
        from .dice_loss import DiceLoss
        from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
        from .subtitle_color_loss import SubtitleColorConsistencyLoss
        
        self.binary_loss_type = binary_loss_type
        self.lambda_color = lambda_color
        self.negative_bce_alpha = negative_bce_alpha
        self.text_binary_threshold = text_binary_threshold
        # YAML에서 문자열로 들어올 수도 있으니 안전하게 float 캐스팅
        self.eps = float(eps)
        
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
        
        # Subtitle mask (full resolution)
        subtitle_mask = (gt.squeeze(1) > 0.5).float()  # (N, H, W)
        
        # Compute scene text mask: text regions that are NOT subtitle
        # This is critical for subtitle-only detection - forces subtitle_binary to suppress scene text
        scene_text_mask = None
        binary_pred = pred.get('binary', None)
        if binary_pred is not None and self.negative_bce_alpha > 0:
            if binary_pred.dim() == 4:
                binary_pred = binary_pred.squeeze(1)  # (N, 1, H, W) -> (N, H, W)
            
            # All text regions from main binary prediction
            text_mask = (binary_pred.detach() > self.text_binary_threshold).float()  # (N, H, W)
            
            # Scene text = text but not subtitle
            scene_text_mask = text_mask * (1.0 - subtitle_mask)  # (N, H, W)
        
        # Update batch with tensor values so color_loss receives tensors
        batch_for_color = batch.copy()
        batch_for_color['gt'] = gt
        batch_for_color['mask'] = mask
        
        # Compute binary loss (subtitle positive supervision)
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
        
        # CRITICAL: Scene text negative loss (forces subtitle_binary to output low values in scene text regions)
        # This is essential for subtitle-only detection - prevents all text from being detected
        negative_loss = None
        subtitle_scene_overlap = None
        if scene_text_mask is not None and self.negative_bce_alpha > 0:
            # Use valid mask (ignore regions) for scene text mask
            valid_scene_text_mask = scene_text_mask * mask  # (N, H, W)
            
            if valid_scene_text_mask.sum() > 0:
                # Target: subtitle_binary should be 0 in scene text regions
                zeros_gt = torch.zeros_like(subtitle_binary)  # (N, 1, H, W)

                # NOTE:
                # BalanceCrossEntropyLoss는 gt가 전부 0인 영역에서는
                # positive_count=0, negative_count=0이 되어 항상 0 loss를 반환한다.
                # 따라서 subtitle-negative 용도에는 맞지 않으므로,
                # 여기서는 픽셀 단위 BCE를 직접 계산해서 사용한다.
                bce_map = F.binary_cross_entropy(
                    subtitle_binary, zeros_gt, reduction='none'
                )[:, 0, :, :]  # (N, H, W)

                masked_bce = bce_map * valid_scene_text_mask  # (N, H, W)
                negative_loss = masked_bce.sum() / (valid_scene_text_mask.sum() + self.eps)

                # Logging용: subtitle_binary가 scene text 영역을 얼마나 subtitle로 잡는지 비율
                with torch.no_grad():
                    subtitle_pred_mask = (subtitle_binary.detach() > 0.5).float().squeeze(1)  # (N, H, W)
                    overlap_pixels = (subtitle_pred_mask * scene_text_mask).sum()
                    total_scene_pixels = scene_text_mask.sum()
                    subtitle_scene_overlap = overlap_pixels / (total_scene_pixels + self.eps)
        
        # Compute color embedding loss (use updated batch with tensors)
        # Include main binary prediction for scene text detection
        # CRITICAL: Use 'subtitle_color_embedding' key to ensure loss supervises subtitle branch
        color_pred = {
            'subtitle_color_embedding': subtitle_color_embedding,  # Explicit key for subtitle branch
            'binary': pred.get('binary', None)  # Main binary for scene text mask calculation
        }
        color_loss, color_metrics = self.color_loss(color_pred, batch_for_color)
        
        # Combine losses
        total_loss = binary_loss + self.lambda_color * color_loss
        if negative_loss is not None:
            total_loss = total_loss + self.negative_bce_alpha * negative_loss
        
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
        
        if negative_loss is not None:
            metrics['subtitle_negative_loss'] = negative_loss.detach()
            metrics['scene_text_pixels'] = scene_text_mask.sum().detach() if scene_text_mask is not None else torch.tensor(0.0, device=device)
            if subtitle_scene_overlap is not None:
                # scene text 영역 중 subtitle_binary가 0.5 이상으로 예측한 비율
                metrics['subtitle_scene_overlap'] = subtitle_scene_overlap.detach()
        
        return total_loss, metrics
