"""
Subtitle-refined loss based on thresh_binary.
Filters loss calculation to actual predicted text regions (not polygon boundaries).
Uses geometric cues (tilt, wobble) to identify subtitle-like regions from thresh_binary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


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
                 binary_threshold=0.5, enable_subtitle_filter=True,
                 enable_color_check=True, color_variance_threshold=0.1,
                 negative_bce_alpha=0.2):
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
        self.enable_color_check = enable_color_check
        self.color_variance_threshold = color_variance_threshold
        # Auxiliary negative BCE on non-subtitle areas
        self.negative_bce_alpha = negative_bce_alpha
    
    def _to_tensor(self, value, device, dtype=torch.float32):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=device, dtype=dtype)
        if isinstance(value, (list, tuple)):
            tensors = [self._to_tensor(v, device, dtype) for v in value]
            return torch.stack(tensors, dim=0)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def compute_subtitle_mask(self, thresh_binary, mask, polygons_list=None, ignore_tags_list=None, images=None):
        """
        Compute subtitle mask from thresh_binary based on geometric cues.
        
        Args:
            thresh_binary: (N, 1, H, W) predicted text regions (픽셀 단위)
            mask: (N, H, W) original mask
            
        Returns:
            subtitle_mask: (N, H, W) mask for subtitle-like regions only
        """
        # Ensure mask is torch tensor on the same device/dtype as thresh_binary
        device = thresh_binary.device if torch.is_tensor(thresh_binary) else torch.device('cpu')
        dtype = thresh_binary.dtype if torch.is_tensor(thresh_binary) else torch.float32
        mask = self._to_tensor(mask, device=device, dtype=dtype)
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask[:, 0]

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

            # If polygons are provided, evaluate per-polygon regions using ROIs
            if polygons_list is not None and b < len(polygons_list) and polygons_list[b] is not None:
                per_sample_mask = torch.zeros_like(orig_mask)
                b_polys = polygons_list[b]
                b_ignores = None
                if ignore_tags_list is not None and b < len(ignore_tags_list):
                    b_ignores = ignore_tags_list[b]

                # polygons are expected as a list of numpy arrays with shape (P, 2)
                for idx, poly in enumerate(b_polys):
                    if b_ignores is not None:
                        try:
                            if int(b_ignores[idx]) != 0:
                                continue
                        except Exception:
                            pass
                    try:
                        # compute bounding box on CPU (lightweight) and clip
                        if not isinstance(poly, np.ndarray):
                            poly = np.array(poly)
                        x_coords = poly[:, 0]
                        y_coords = poly[:, 1]
                        xmin = int(max(0, min(float(x_coords.min()), W - 1)))
                        xmax = int(min(W - 1, max(float(x_coords.max()), 0)))
                        ymin = int(max(0, min(float(y_coords.min()), H - 1)))
                        ymax = int(min(H - 1, max(float(y_coords.max()), 0)))
                    except Exception:
                        # Fallback: skip malformed polygon
                        continue

                    if xmax < xmin or ymax < ymin:
                        continue

                    roi = text_map[ymin:ymax + 1, xmin:xmax + 1]                                    # thresh_binary를 이진화한 text_map에서 해당 폴리곤의 bbox 영역만 잘라낸 예측 맵(torch, float)
                    if roi.numel() == 0:
                        continue

                    # Rasterize polygon inside ROI to mask strictly within polygon
                    try:
                        poly_rel = poly.copy()                                                      # poly 좌표를 ROI 좌표계로 변환
                        poly_rel[:, 0] = poly_rel[:, 0] - xmin
                        poly_rel[:, 1] = poly_rel[:, 1] - ymin
                        h_roi = ymax - ymin + 1
                        w_roi = xmax - xmin + 1
                        poly_mask_np = np.zeros((h_roi, w_roi), dtype=np.uint8)
                        cv2.fillPoly(poly_mask_np, [poly_rel.astype(np.int32)], 1)                  # ROI 내 polygon 내부 픽셀을 1로 채움
                        poly_mask = torch.from_numpy(poly_mask_np).to(roi.device, dtype=roi.dtype)
                        roi = roi * poly_mask                                                       # ROI 내 polygon 내부 픽셀만 남김
                    except Exception:
                        # If rasterization fails, proceed with bbox-only mask
                        pass

                    if roi.sum() < 1:
                        continue

                    roi_sub = self._check_flatness_gpu(roi)

                    # Optional color consistency check inside predicted region
                    if self.enable_color_check and images is not None and roi_sub.sum() > 0:
                        # Extract corresponding image ROI (C, H_roi, W_roi)
                        try:
                            img_roi = images[b][:, ymin:ymax + 1, xmin:xmax + 1]
                            # Use only pixels predicted as subtitle-like (roi_sub>0)
                            region_ok = self._check_color_consistency_gpu(img_roi, roi_sub)
                            if not region_ok:
                                continue
                        except Exception:
                            # If any issue occurs, skip color check
                            pass
                    # Merge back into per-sample mask
                    per_sample_mask[ymin:ymax + 1, xmin:xmax + 1] = torch.maximum(
                        per_sample_mask[ymin:ymax + 1, xmin:xmax + 1], roi_sub)

                subtitle_map = per_sample_mask
            else:
                # Global check: apply geometric constraints on the whole map
                # Apply original mask
                text_map = text_map * orig_mask
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

    def _check_color_consistency_gpu(self, img_roi, region_mask):
        """
        Check color consistency (dominant color presence) within region.

        Args:
            img_roi: (C, H, W) image patch tensor (already on device)
            region_mask: (H, W) float/binary tensor where >0 indicates region

        Returns:
            bool: True if color variance is below threshold (consistent color)
        """
        # Ensure float operations
        if img_roi.dtype != torch.float32:
            img_roi = img_roi.float()
        mask = (region_mask > 0).float()
        num = mask.sum()
        if num.item() < 10:
            return False
        # Expand mask to channels
        mask_c = mask.unsqueeze(0)
        # Compute per-channel mean and variance under mask
        # E[X]
        mean = (img_roi * mask_c).sum(dim=(1, 2)) / (num + 1e-6)
        # E[X^2]
        mean_sq = ((img_roi ** 2) * mask_c).sum(dim=(1, 2)) / (num + 1e-6)
        var = torch.clamp(mean_sq - mean ** 2, min=0.0)
        avg_variance = var.mean().item()
        return avg_variance < self.color_variance_threshold
    
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
            device = pred['binary'].device
            dtype = pred['binary'].dtype
            mask_tensor = self._to_tensor(batch['mask'], device=device, dtype=dtype)
            gt_tensor = self._to_tensor(batch['gt'], device=device, dtype=dtype)
            thresh_mask_tensor = None
            if 'thresh_mask' in batch and batch['thresh_mask'] is not None:
                thresh_mask_tensor = self._to_tensor(batch['thresh_mask'], device=device, dtype=dtype)
            else:
                thresh_mask_tensor = mask_tensor

            subtitle_mask = self.compute_subtitle_mask(
                pred['thresh_binary'], mask_tensor,
                polygons_list=batch.get('polygons', None),
                ignore_tags_list=batch.get('ignore_tags', None),
                images=batch.get('image', None))
            
            # Apply subtitle mask to loss calculations
            # Combine with original mask (both must be 1)
            combined_mask = subtitle_mask * mask_tensor
            combined_thresh_mask = subtitle_mask * thresh_mask_tensor if thresh_mask_tensor is not None else combined_mask
            
            # BCE는 원래 mask 사용, Dice/L1만 subtitle 필터 적용
            bce_loss = self.bce_loss(pred['binary'], gt_tensor, mask_tensor)
            l1_loss, l1_metric = self.l1_loss(
                pred['thresh'], batch['thresh_map'], combined_thresh_mask)
            dice_loss = self.dice_loss(
                pred['thresh_binary'], gt_tensor, combined_mask)
            # Auxiliary negative BCE on non-subtitle regions
            bce_neg = None
            if self.negative_bce_alpha is not None and self.negative_bce_alpha > 0:
                non_sub_mask = (mask_tensor * (1.0 - subtitle_mask)).clamp(min=0.0, max=1.0)
                zeros_gt = torch.zeros_like(gt_tensor)
                bce_neg = self.bce_loss(pred['binary'], zeros_gt, non_sub_mask)
            
            metrics = dict(bce_loss=bce_loss)
            metrics['thresh_loss'] = dice_loss
            subtitle_coverage = combined_mask.sum() / (mask_tensor.sum() + 1e-8)
            metrics['subtitle_coverage'] = subtitle_coverage.detach()
            if bce_neg is not None:
                metrics['bce_neg'] = bce_neg
            
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
            if bce_neg is not None:
                loss = loss + self.negative_bce_alpha * bce_neg
            metrics.update(**l1_metric)
        else:
            # Fallback: no subtitle filtering if thresh_binary not available
            mask_tensor = self._to_tensor(batch['mask'], device=device, dtype=dtype)
            gt_tensor = self._to_tensor(batch['gt'], device=device, dtype=dtype)
            bce_loss = self.bce_loss(pred['binary'], gt_tensor, mask_tensor)
            metrics = dict(bce_loss=bce_loss)
            loss = bce_loss
        
        return loss, metrics
