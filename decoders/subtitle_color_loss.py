import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class SubtitleColorConsistencyLoss(nn.Module):
    """
    Minimal subtitle-only color embedding loss.
    
    Core components (essential for subtitle-only detection):
    1. **Intra-subtitle compactness**: Each subtitle instance forms a tight cluster
    2. **Subtitle vs scene text negative**: Push scene text away from subtitle cluster (critical)
    
    Optional component:
    3. **Instance consistency**: Multiple subtitle instances share same style (optional)
    
    Removed components (not needed for subtitle-only):
    - Subtitle vs background separation (background is not text, scene text negative is sufficient)
    - Inter-subtitle instance vs background (same reason)
    """

    def __init__(self,
                 lambda_instance_consistency: float = 0.1,  # Optional: can be 0.0 to disable
                 lambda_subtitle_vs_scene_text: float = 0.5,  # Essential: scene text negative
                 margin: float = 0.5,
                 binary_threshold: float = 0.5,
                 text_binary_threshold: float = 0.5,
                 min_pixels: int = 20,
                 eps: float = 1e-6):
        super().__init__()
        self.lambda_instance_consistency = lambda_instance_consistency
        self.lambda_subtitle_vs_scene_text = lambda_subtitle_vs_scene_text
        self.margin = margin
        self.min_pixels = min_pixels
        self.binary_threshold = binary_threshold
        self.text_binary_threshold = text_binary_threshold
        self.eps = eps

    def forward(self, pred, batch):
        # Use subtitle_color_embedding (NOT general color_embedding)
        # This is critical: loss must supervise subtitle branch, not general branch
        if 'subtitle_color_embedding' in pred:
            color = pred['subtitle_color_embedding']  # (N, C, H, W) - Subtitle branch embedding
        elif 'color_embedding' in pred:
            # Fallback for backward compatibility (should not be used for subtitle branch)
            color = pred['color_embedding']  # (N, C, H, W) - General branch embedding
        else:
            raise KeyError(
                "Neither 'subtitle_color_embedding' nor 'color_embedding' found in pred. "
                "SubtitleColorConsistencyLoss requires subtitle_color_embedding for subtitle branch supervision."
            )
        gt = batch['gt']                         # (N, 1, H, W)

        # Subtitle mask: GT-based (full resolution)
        subtitle_mask_full = (gt > self.binary_threshold).float().squeeze(1)  # (N, H, W) - full resolution

        # All text mask from model binary pred (full resolution)
        binary_pred = pred.get('binary', None)
        scene_text_mask_full = None
        if binary_pred is not None:
            if binary_pred.dim() == 4:
                binary_pred = binary_pred.squeeze(1)
            
            # STABLE threshold
            text_mask = (binary_pred.detach() > self.text_binary_threshold).float()

            # scene text = text but not subtitle (full resolution에서 계산)
            scene_text_mask_full = text_mask * (1.0 - subtitle_mask_full)
        
        # CRITICAL: Downsample masks to match feature resolution (1/4)
        # subtitle_color_embedding is 1/4 resolution, so masks must be downsampled
        # Use nearest mode to preserve binary mask values (0 or 1)
        target_size = (color.shape[-2], color.shape[-1])  # (H/4, W/4)
        
        subtitle_mask = F.interpolate(
            subtitle_mask_full.unsqueeze(1),  # (N, H, W) -> (N, 1, H, W)
            size=target_size,
            mode='nearest',
            align_corners=None
        ).squeeze(1)  # (N, 1, H/4, W/4) -> (N, H/4, W/4)
        
        scene_text_mask = None
        if scene_text_mask_full is not None:
            scene_text_mask = F.interpolate(
                scene_text_mask_full.unsqueeze(1),  # (N, H, W) -> (N, 1, H, W)
                size=target_size,
                mode='nearest',
                align_corners=None
            ).squeeze(1)  # (N, 1, H/4, W/4) -> (N, H/4, W/4)

        # bg_mask no longer needed (removed background separation loss)
        return self._compute_loss(color, subtitle_mask, scene_text_mask)


    # --------------------
    # Utilities
    # --------------------

    def _masked_mean(self, feat, mask):
        denom = mask.sum() + self.eps
        return (feat * mask.unsqueeze(0)).sum(dim=(1,2)) / denom

    def _extract_components(self, mask):
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_np, connectivity=8)

        comps = []
        for i in range(1, num_labels):
            comp = (labels == i).astype(np.float32)
            if comp.sum() < self.min_pixels:
                continue
            comps.append(torch.from_numpy(comp))
        return comps


    # --------------------
    # Main Loss
    # --------------------

    def _compute_loss(self, color, subtitle_mask, scene_text_mask=None):
        N, C, H, W = color.shape

        total = 0
        intra_sum = color.new_zeros(())
        inst_consistency_sum = color.new_zeros(())
        sub_vs_scene_sum = color.new_zeros(())

        inst_consistency_n = 0
        sub_vs_scene_n = 0

        for b in range(N):
            feat = color[b]
            sub_mask = subtitle_mask[b]

            # Skip if no subtitle
            if sub_mask.sum() < self.min_pixels:
                continue

            comps = self._extract_components(sub_mask)
            if len(comps) == 0:
                continue

            # global subtitle style
            m_sub_global = self._masked_mean(feat, sub_mask)

            # scene text center (explicit negative - essential for subtitle-only)
            m_scene = None
            if scene_text_mask is not None:
                scene = scene_text_mask[b]
                if scene.sum() >= self.min_pixels:
                    m_scene = self._masked_mean(feat, scene)

            # ---- Per subtitle instance ----
            instance_means = []
            for comp in comps:
                comp = comp.to(feat.device)

                if comp.sum() < self.min_pixels:
                    continue

                m_inst = self._masked_mean(feat, comp)
                instance_means.append(m_inst)

                # 1) Intra compactness (ESSENTIAL: tight subtitle cluster)
                delta = feat - m_inst.view(C, 1, 1)
                dist_sq = (delta ** 2).sum(dim=0)
                intra = (dist_sq * comp).sum() / (comp.sum() + self.eps)
                intra_sum += intra

                total += 1

            # 2) Instance consistency (OPTIONAL: style alignment for multiple instances)
            if len(instance_means) > 1:
                for m_inst in instance_means:
                    dist = torch.norm(m_inst - m_sub_global, p=2)
                    inst_consistency_sum += dist ** 2
                    inst_consistency_n += 1

            # 3) Subtitle vs SCENE TEXT separation (ESSENTIAL: subtitle-only detector)
            if m_scene is not None and sub_mask.sum() >= self.min_pixels:
                dist_scene = torch.norm(m_sub_global - m_scene, p=2)
                # Use stronger margin (2.0x) to push scene text far away from subtitle cluster
                sub_scene_loss = F.relu(self.margin * 2.0 - dist_scene) ** 2
                sub_vs_scene_sum += sub_scene_loss
                sub_vs_scene_n += 1

        # ----------- Averages -------------
        if total == 0:
            zero = color.new_zeros(())
            return zero, {
                'color_loss': zero,
                'color_intra': zero,
                'color_inst': zero,
                'color_sub_vs_scene': zero,
            }

        intra_avg = intra_sum / total

        inst_consistency_avg = (
            inst_consistency_sum / inst_consistency_n if inst_consistency_n > 0 else color.new_zeros(())
        )

        sub_vs_scene_avg = (
            sub_vs_scene_sum / sub_vs_scene_n if sub_vs_scene_n > 0 else color.new_zeros(())
        )

        # ----------- Total Loss ----------
        # Minimal loss: only essential components for subtitle-only detection
        loss = (
            intra_avg  # Essential: tight subtitle cluster
            + self.lambda_instance_consistency * inst_consistency_avg  # Optional: style alignment
            + self.lambda_subtitle_vs_scene_text * sub_vs_scene_avg  # Essential: scene text negative
        )

        return loss, {
            'color_loss': loss.detach(),
            'color_intra': intra_avg.detach(),
            'color_inst': inst_consistency_avg.detach(),
            'color_sub_vs_scene': sub_vs_scene_avg.detach(),
            # Debug info
            'color_sub_vs_scene_count': color.new_tensor(float(sub_vs_scene_n)),
        }
