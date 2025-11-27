import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    """
    Lightweight domain adaptation layer on top of the highest-level backbone feature (F5).

    This module takes the backbone's top feature map (e.g., ResNet C5) and slightly
    shifts it toward the subtitle domain without touching the original DBNet text
    detection path.
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, f5: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f5: Top-level backbone feature map, shape (N, C_in, H, W).

        Returns:
            F_sub: Subtitle-adapted feature map, shape (N, C_out, H, W).
        """
        return self.adapter(f5)


class SubtitleStyleHead(nn.Module):
    """
    Subtitle Style Head built on top of the adapted F5 feature.

    Responsibilities:
      1) Predict a subtitle likelihood map S (subtitle vs non-subtitle) from F_sub.
      2) Produce a pixel-level style embedding map E_map used for style consistency.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 32,
        subtitle_head_channels: int = 128,
    ):
        super().__init__()

        # Small conv stack before predictions (kept lightweight).
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, subtitle_head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(subtitle_head_channels),
            nn.ReLU(inplace=True),
        )

        # Subtitle likelihood head: 1-channel score map → sigmoid outside if needed.
        self.subtitle_logits = nn.Conv2d(subtitle_head_channels, 1, kernel_size=1, bias=True)

        # Style embedding head: D-channel pixel-level embedding.
        self.embedding_conv = nn.Conv2d(subtitle_head_channels, embed_dim, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 1e-4)

    def forward(self, f_sub: torch.Tensor):
        """
        Args:
            f_sub: Adapted top-level feature, shape (N, C_in, H, W).

        Returns:
            A dict with:
              - 'subtitle_s': subtitle probability map at F5 resolution (sigmoid applied)
              - 'subtitle_embedding': pixel-level style embedding map E_map
        """
        x = self.trunk(f_sub)
        logits = self.subtitle_logits(x)
        s_map = torch.sigmoid(logits)
        e_map = self.embedding_conv(x)
        return {
            "subtitle_s": s_map,
            "subtitle_embedding": e_map,
        }


