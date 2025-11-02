"""
Subtitle-refined loss based on thresh_binary.
Filters loss calculation to actual predicted text regions (not polygon boundaries).
Uses geometric cues (tilt, wobble) to identify subtitle-like regions from thresh_binary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubtitleRefinedL1BalanceCELoss(nn.Module):
    """
    L1BalanceCELoss with subtitle filtering based on thresh_binary.
    
    Filters loss calculation to subtitle-like regions by:
    1. Extracting actual text regions from thresh_binary (픽셀 단위 예측)
    2. Checking geometric properties (tilt, wobble) using tensor operations
    3. Masking non-subtitle regions to exclude them from loss
    
    This is more accurate than data loading stage filtering because:
    - Uses actual predicted text regions (thresh_binary) instead of polygon bounding boxes
    - Pixel-level precision for geometric checks
    - Self-refinement: improves as training progresses
    """
    
    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5,
                 tilt_threshold=0.3, wobble_threshold=5.0,
                 binary_threshold=0.5, enable_subtitle_filter=True):
        super(SubtitleRefinedL1BalanceCELoss, self).__init__()
        from .dice_loss import DiceLoss
        from .l1_loss import MaskL1Loss
        from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
        
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()
        
        self.l1_scale = l1_scale
        self.bce_scale = bce_scale
        
        # Subtitle filtering parameters
        self.tilt_threshold = tilt_threshold
        self.wobble_threshold = wobble_threshold
        self.binary_threshold = binary_threshold
        self.enable_subtitle_filter = enable_subtitle_filter
    
    def compute_subtitle_mask(self, thresh_binary, mask):
        """
        Compute subtitle mask from thresh_binary based on geometric cues.
        
        Args:
            thresh_binary: (N, 1, H, W) predicted text regions (픽셀 단위)
            mask: (N, H, W) original mask
            
        Returns:
            subtitle_mask: (N, H, W) mask for subtitle-like regions only
        """
        if not self.enable_subtitle_filter or thresh_binary is None:
            return mask
        
        N, C, H, W = thresh_binary.shape
        
        # Binarize thresh_binary to get actual text regions
        # Detach to prevent gradient flow through mask computation
        binary_map = (thresh_binary.detach() > self.binary_threshold).float()
        
        # Ensure mask has correct shape (N, H, W)
        if mask.dim() == 2:
            # Single sample, expand to (1, H, W)
            mask = mask.unsqueeze(0)
        if mask.dim() == 3 and mask.shape[0] == 1:
            # Expand to (N, H, W)
            mask = mask.expand(N, -1, -1)
        
        # Process each sample in batch
        subtitle_masks = []
        for b in range(N):
            text_map = binary_map[b, 0]  # (H, W)
            orig_mask = mask[b] if mask.dim() == 3 else mask  # (H, W)
            
            # Apply original mask
            text_map = text_map * orig_mask
            
            # Check tilt and wobble for subtitle-like regions
            subtitle_map = self._check_flatness_gpu(text_map)
            
            # Combine with original mask
            subtitle_mask = subtitle_map * orig_mask
            subtitle_masks.append(subtitle_mask)
        
        subtitle_mask = torch.stack(subtitle_masks, dim=0)  # (N, H, W)
        return subtitle_mask
    
    def _check_flatness_gpu(self, text_map):
        """
        Check flatness (tilt and wobble) of text regions on GPU.
        Uses tensor operations for efficiency.
        
        Args:
            text_map: (H, W) binary map of text regions (실제 예측된 텍스트 영역)
            
        Returns:
            subtitle_map: (H, W) mask for subtitle-like regions (1 for subtitle, 0 otherwise)
        """
        H, W = text_map.shape
        
        # Compute y-coordinate counts for each x position
        y_counts_per_x = text_map.sum(dim=0)  # (W,) sum along y-axis for each x
        
        # Filter out x positions with zero y-coordinates
        active_x_mask = (y_counts_per_x > 0).float()  # (W,)
        num_active = active_x_mask.sum().item()
        
        if num_active < 2:
            # Not enough active x positions
            return torch.zeros_like(text_map)
        
        # Get y_counts for active x positions only
        y_counts_active = y_counts_per_x[active_x_mask > 0]  # (num_active,)
        
        # Check 1: Tilt (y-count uniformity)
        mean_y_count = y_counts_active.mean()
        std_y_count = y_counts_active.std()
        
        if mean_y_count > 1e-6:  # Avoid division by zero
            cv_y_count = std_y_count / mean_y_count  # Coefficient of variation
            if cv_y_count > self.tilt_threshold:
                # Too tilted/rotated
                return torch.zeros_like(text_map)
        
        # Check 2: Wobble (centerline flatness)
        # Compute centerline Y coordinates for each x position
        centerline_y_list = []
        
        # Vectorized computation of centerline
        for x in range(W):
            if active_x_mask[x] > 0:
                y_active = torch.nonzero(text_map[:, x] > 0, as_tuple=False)
                if len(y_active) > 0:
                    center_y = (y_active.min().float() + y_active.max().float()) / 2.0
                    centerline_y_list.append(center_y)
        
        if len(centerline_y_list) < 2:
            return torch.zeros_like(text_map)
        
        centerline_y_tensor = torch.stack(centerline_y_list)
        std_centerline = centerline_y_tensor.std().item()
        
        if std_centerline > self.wobble_threshold:
            # Too wobbly
            return torch.zeros_like(text_map)
        
        # Both checks passed - return original text_map as subtitle region
        return text_map
    
    def forward(self, pred, batch):
        """
        Forward pass with subtitle filtering.
        
        Args:
            pred: dict containing 'binary', 'thresh', 'thresh_binary'
            batch: dict containing 'gt', 'mask', 'thresh_map', 'thresh_mask'
            
        Returns:
            loss: scalar loss value
            metrics: dict containing partial loss values
        """
        if 'thresh' in pred and 'thresh_binary' in pred:
            # Compute subtitle mask from thresh_binary (pred 기반 필터링)
            subtitle_mask = self.compute_subtitle_mask(
                pred['thresh_binary'], batch['mask'])
            
            # Apply subtitle mask to loss calculations
            # Combine with original mask (both must be 1)
            combined_mask = subtitle_mask * batch['mask']
            combined_thresh_mask = subtitle_mask * batch['thresh_mask'] if 'thresh_mask' in batch else combined_mask
            
            # All losses use subtitle filtering
            bce_loss = self.bce_loss(pred['binary'], batch['gt'], combined_mask)
            l1_loss, l1_metric = self.l1_loss(
                pred['thresh'], batch['thresh_map'], combined_thresh_mask)
            dice_loss = self.dice_loss(
                pred['thresh_binary'], batch['gt'], combined_mask)
            
            metrics = dict(bce_loss=bce_loss)
            metrics['thresh_loss'] = dice_loss
            metrics['subtitle_coverage'] = (combined_mask.sum() / (batch['mask'].sum() + 1e-8)).item()
            
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
            metrics.update(**l1_metric)
        else:
            # Fallback: no subtitle filtering if thresh_binary not available
            bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
            metrics = dict(bce_loss=bce_loss)
            loss = bce_loss
        
        return loss, metrics
