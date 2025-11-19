from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubtitleColorConsistencyLoss(nn.Module):
    """
    Self-refined color embedding consistency loss.

    Encourages subtitle embeddings to form compact clusters while pushing them
    away from background embeddings. Also supports optional mask refinement
    using embedding distance to improve pseudo-label quality.
    """

    def __init__(self,
                 lambda_inter: float = 0.4,
                 margin: float = 0.5,
                 refine_sigma: float = 0.25,
                 refine_threshold: Optional[float] = 0.2,
                 min_pixels: int = 20,
                 binary_threshold: float = 0.5,
                 use_binary_mask: bool = True,
                 use_subtitle_mask: bool = False,   # @@
                 gt_threshold: float = 0.5,
                 eps: float = 1e-6):
        super().__init__()
        self.lambda_inter = lambda_inter
        self.margin = margin
        self.refine_sigma = refine_sigma
        self.refine_threshold = refine_threshold
        self.min_pixels = min_pixels
        self.binary_threshold = binary_threshold
        self.use_binary_mask = use_binary_mask
        self.use_subtitle_mask = use_subtitle_mask
        self.gt_threshold = gt_threshold
        self.eps = eps

    def forward(self, pred, batch):
        if not isinstance(pred, dict):
            raise TypeError("SubtitleColorConsistencyLoss expects prediction dict as first argument.")

        color_embedding = pred.get('color_embedding', None)
        if color_embedding is None:
            raise ValueError("Predictions must include 'color_embedding' for SubtitleColorConsistencyLoss.")

        device = color_embedding.device
        dtype = color_embedding.dtype

        # batch['gt'], batch['mask']를 텐서로 변환 ********************************************************
        gt_tensor = self._to_tensor(batch.get('gt'), device=device, dtype=dtype)                           
        if gt_tensor is None:
            raise ValueError("Batch must include 'gt' tensor for SubtitleColorConsistencyLoss.")

        mask_tensor = self._to_tensor(batch.get('mask'), device=device, dtype=dtype)
        if mask_tensor is None:
            mask_tensor = torch.ones(
                color_embedding.size(0),
                color_embedding.size(-2),
                color_embedding.size(-1),
                device=device,
                dtype=dtype)
        #  **********************************************************************************************

        if mask_tensor.dim() == 4 and mask_tensor.size(1) == 1:
            mask_tensor = mask_tensor[:, 0]
        elif mask_tensor.dim() == 3:
            mask_tensor = mask_tensor
        else:
            raise ValueError("Mask tensor must have shape (N,H,W) or (N,1,H,W).")

        binary_pred = pred.get('binary', None)
        subtitle_mask = pred.get('thresh_binary', None) if self.use_subtitle_mask else None
        if subtitle_mask is not None:
            subtitle_mask = subtitle_mask.detach()                                                          # subtitle_mask에 들어가 있는 thresh_binary 값은 color loss gradient 대상에서 제외

        # loss 계산
        loss, metrics = self._compute_loss(
            color_embedding=color_embedding,
            binary_pred=binary_pred,
            gt_tensor=gt_tensor,
            mask_tensor=mask_tensor,
            subtitle_mask=subtitle_mask)
        return loss, metrics

    def _to_tensor(self, value, device, dtype):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)


    def _compute_loss(self,
                      color_embedding: torch.Tensor,
                      binary_pred: Optional[torch.Tensor] = None,
                      gt_tensor: Optional[torch.Tensor] = None,
                      mask_tensor: Optional[torch.Tensor] = None,
                      subtitle_mask: Optional[torch.Tensor] = None):
        """
        Compute color consistency loss.

        Args:
            color_embedding: (N, D, H, W) normalized embedding map.
            binary_pred: (N, 1, H, W) predicted binary map.
            gt_tensor: (N, 1, H, W) ground-truth shrink mask.
            mask_tensor: (N, H, W) or (N, 1, H, W) ignore mask.
            subtitle_mask: (N, H, W) refined subtitle mask (optional).

        Returns:
            Tuple (loss, metrics dict). If no valid components, returns (None, {}).
        """
        # validate input value (color_embedding, gt_tensor, mask_tensor)
        if color_embedding is None:
            raise ValueError("color_embedding tensor is required for SubtitleColorConsistencyLoss.")
        if gt_tensor is None or mask_tensor is None:
            zero = color_embedding.new_zeros(())
            metrics = {'color_components': color_embedding.new_tensor(0.0)}
            return zero, metrics

        device = color_embedding.device
        dtype = color_embedding.dtype

        gt_mask = (gt_tensor > self.gt_threshold).float()                   # 0 또는 1
        if gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(1)

        if mask_tensor.dim() == 4 and mask_tensor.size(1) == 1:
            mask_tensor = mask_tensor[:, 0]
        mask_tensor = mask_tensor.float()
        valid_mask = (mask_tensor > 0.5).float()                            # 0 또는 1

        base_mask = gt_mask.clone()
        base_mask_sum_before = base_mask.sum().item()
        valid_mask_sum = valid_mask.sum().item()
        #print(f"[DEBUG][SubtitleColorLoss] base_mask_before={base_mask_sum_before:.1f}, valid_mask={valid_mask_sum:.1f}")

        binary_mask = None
        if self.use_binary_mask and binary_pred is not None:
            binary_mask = (binary_pred.detach() > self.binary_threshold).float()    # 여기서의 값이 binary_pred에 영향을 주면 안됨
            if binary_mask.dim() == 4:
                binary_mask = binary_mask.squeeze(1)
            #print(f"[DEBUG][SubtitleColorLoss] binary_mask_sum={binary_mask.sum().item():.1f}")
            base_mask = torch.maximum(base_mask, binary_mask)                       # GT shrink mask, binary mask

        if self.use_subtitle_mask and subtitle_mask is not None:
            if subtitle_mask.dim() == 4 and subtitle_mask.size(1) == 1:
                subtitle_mask = subtitle_mask[:, 0]
            subtitle_mask = subtitle_mask.to(device=device, dtype=base_mask.dtype)
            base_mask = torch.maximum(base_mask, subtitle_mask)                     # GT shrink mask, binary mask, subtitle mask 

        base_mask = (base_mask > 0).float() * valid_mask                            # initialize mask, 학습에서 무시할 영역 제외
        background_mask = (valid_mask - base_mask).clamp(min=0.0)                   # background mask, 학습에서 무시할 영역 제외

        base_mask_sum_after = base_mask.sum().item()
        #print(f"[DEBUG][SubtitleColorLoss] base_mask_after={base_mask_sum_after:.1f}")
        
        total_components = 0
        intra_accum = color_embedding.new_zeros(())                        
        inter_accum = color_embedding.new_zeros(())

        batch_size = color_embedding.shape[0]
        for b in range(batch_size):
            sample_mask = base_mask[b]
            if sample_mask.sum() <= self.min_pixels:
                continue

            components = self._extract_connected_components(sample_mask, self.min_pixels)
            if len(components) == 0:
                continue

            embedding_b = color_embedding[b]
            bg_mask_b = background_mask[b]
            mu_bg = None
            bg_count = bg_mask_b.sum().item()
            #if torch.isnan(bg_mask_b).any():
                #print(f"[DEBUG][SubtitleColorLoss] NaN detected in background mask at sample {b}")
            #print(f"[DEBUG][SubtitleColorLoss] sample={b}, background_pixels={bg_count:.1f}")
            # 먼저 background mean을 계산
            if bg_count > self.min_pixels:
                mu_bg = self._masked_mean(embedding_b, bg_mask_b)
            else:
                print(f"[DEBUG][SubtitleColorLoss] skip inter (background too small, min={self.min_pixels})")
            valid_mask_b = valid_mask[b]

            for comp_mask in components:
                comp_mask = comp_mask.to(device=device, dtype=dtype)
                comp_mask = comp_mask * valid_mask_b
                if comp_mask.sum() <= self.min_pixels:
                    continue

                mu_sub = self._masked_mean(embedding_b, comp_mask)
                weights = self._refine_weights(embedding_b, mu_sub, comp_mask)      # 각 픽셀의 feature weight와 얼마나 가까운지에 따라 weight 계산
                if weights.sum() <= self.min_pixels:
                    weights = comp_mask

                intra_loss = self._compute_intra_loss(embedding_b, mu_sub, weights)
                intra_accum = intra_accum + intra_loss

                if mu_bg is not None:
                    dist = torch.norm(mu_sub - mu_bg, p=2)
                    inter_loss = F.relu(self.margin - dist) ** 2
                    inter_accum = inter_accum + inter_loss
                total_components += 1

        if total_components == 0:
            zero = color_embedding.new_zeros(())
            metrics = {
                'color_loss': zero.detach(),
                'color_intra': zero.detach(),
                'color_inter': zero.detach(),
                'color_components': color_embedding.new_tensor(0.0),
            }
            return zero, metrics

        intra_loss = intra_accum / total_components
        inter_loss = inter_accum / max(total_components, 1)
        color_loss = intra_loss + self.lambda_inter * inter_loss

        metrics = {
            'color_loss': color_loss.detach(),
            'color_intra': intra_loss.detach(),
            'color_inter': inter_loss.detach(),
            'color_components': color_embedding.new_tensor(float(total_components))
        }
        return color_loss, metrics

    def _extract_connected_components(self, mask: torch.Tensor, min_pixels: int):
        if mask.sum().item() <= 0:
            return []
        mask_np = mask.detach().cpu().numpy()
        mask_bin = (mask_np > 0.5).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_bin, connectivity=8)
        components = []
        for label_idx in range(1, num_labels):
            component_np = (labels == label_idx)
            if component_np.sum() < min_pixels:
                continue
            component_tensor = torch.from_numpy(component_np.astype(np.float32)).to(mask.device)
            components.append(component_tensor)
        return components

    def _masked_mean(self, embedding: torch.Tensor, mask: torch.Tensor):
        mask = mask.float()
        denom = mask.sum() + self.eps
        return (embedding * mask.unsqueeze(0)).sum(dim=(1, 2)) / denom

    def _compute_intra_loss(self, embedding: torch.Tensor, mu: torch.Tensor, weights: torch.Tensor):
        weights = weights.float()
        weight_sum = weights.sum() + self.eps
        if weight_sum <= self.eps:
            return embedding.new_zeros(())
        delta = embedding - mu.view(-1, 1, 1)
        squared_dist = (delta ** 2).sum(dim=0)
        return (squared_dist * weights).sum() / weight_sum

    def _refine_weights(self, embedding: torch.Tensor, mu: torch.Tensor, base_mask: torch.Tensor):
        base_mask = base_mask.float()
        if base_mask.sum() <= self.eps:
            return base_mask
        delta = embedding - mu.view(-1, 1, 1)
        distance = torch.norm(delta, dim=0)
        sigma = max(self.refine_sigma, self.eps)
        weights = torch.exp(-distance / sigma) * base_mask
        if self.refine_threshold is not None:
            weights = torch.where(weights >= self.refine_threshold,
                                  weights, torch.zeros_like(weights))
        return weights

