import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

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
                 binary_threshold: float = 0.5,
                 min_pixels: int = 20,
                 eps: float = 1e-6):
        super().__init__()
        self.lambda_inter = lambda_inter
        self.margin = margin
        self.min_pixels = min_pixels
        self.binary_threshold = binary_threshold
        self.eps = eps

    def forward(self, pred, batch):
        color = pred['color_embedding']          # (N, 16, H, W)
        gt = batch['gt']                         # (N, 1, H, W)
        
        subtitle_mask = (gt > self.binary_threshold).float()
        subtitle_mask = subtitle_mask.squeeze(1)            # (N,H,W)

        bg_mask = 1.0 - subtitle_mask                       # (N,H,W)
        return self._compute_loss(color, subtitle_mask, bg_mask)



# -----------------------
# Utilities

    def _masked_mean(self, feat, mask):
        denom = mask.sum() + self.eps
        return (feat * mask.unsqueeze(0)).sum(dim=(1,2)) / denom
    
    
    def _extract_components(self, mask):
        """
        mask: (H,W) float {0,1}
        returns: list of torch.FloatTensor masks (H,W)
        """
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_np, connectivity=8)

        comps = []
        for i in range(1, num_labels):
            comp = (labels == i).astype(np.float32)
            if comp.sum() < self.min_pixels:
                continue
            comps.append(torch.from_numpy(comp))
        return comps

# -----------------------
# Main loss computation
    def _compute_loss(self, color, subtitle_mask, bg_mask):
        N, C, H, W = color.shape
        total = 0
        intra_sum = color.new_zeros(())
        inter_sum = color.new_zeros(())

        for b in range(N):
            feat = color[b]
            sub_mask = subtitle_mask[b]
            bg = bg_mask[b]

            if bg.sum() < self.min_pixels:
                continue

            m_bg = self._masked_mean(feat, bg)

            # intra loss 
            comps = self._extract_components(sub_mask)
            for comp in comps:
                comp = comp.to(feat.device)

                if comp.sum() < self.min_pixels:
                    continue

                m_sub = self._masked_mean(feat, comp)
                delta = feat - m_sub.view(C, 1, 1)
                dist_sq = (delta ** 2).sum(dim=0)
                intra = (dist_sq * comp).sum() / (comp.sum() + self.eps)

                # inter loss
                dist = torch.norm(m_sub - m_bg, p=2)
                inter = F.relu(self.margin - dist) ** 2

                intra_sum += intra
                inter_sum += inter
                total += 1

        if total == 0:
            zero = color.new_zeros(())
            metrics = {
                'color_loss': zero,
                'color_intra': zero,
                'color_inter': zero,
                'components': color.new_tensor(0.0)
            }
            return zero, metrics
        
        intra_avg = intra_sum / total
        inter_avg = inter_sum / total
        loss = intra_avg + self.lambda_inter * inter_avg

        metrics = {
            'color_loss': loss,
            'color_intra': intra_avg.detach(),
            'color_inter': inter_avg.detach(),
            'components': color.new_tensor(float(total))
        }

        return loss, metrics
