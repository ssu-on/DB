import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorEmbeddingHead(nn.Module):
    """
    Color embedding prediction head.

    Projects fused multi-scale features into a low-dimensional embedding space
    that captures subtitle-specific color characteristics.
    
    NOTE: This version uses upsampling (ConvTranspose2d), which can restore
    high-frequency scene text edges. For subtitle-only detection, use
    SubtitleColorEmbeddingHead instead (no upsampling).
    """

    # 입력 채널 수, 중간 채널 비율
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 16,
        hidden_ratio: int = 4,
        bias: bool = False,
        normalize: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        hidden_channels = max(in_channels // hidden_ratio, embed_dim)                                   # @@ 입력 채널을 hidden_ratio로 나눈 값을 기본 중간 채널 수로 쓰되, 지나치게 작아지지 않도록 최소값을 embed_dim으로 보장
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, embed_dim, kernel_size=2, stride=2, bias=True),
        )
        self.normalize = normalize
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.projection(x)                              
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1, eps=self.eps)                                # 각 픽셀 위치의 embedding vector 길이를 1로 맞추는 작업. 학습이 방향(코사인) 중심으로 진행되서, eature magnitude에 덜 민감하고 안정적인 거리/유사도 계산이 가능.
        return embedding


class SubtitleColorEmbeddingHead(nn.Module):
    """
    Subtitle-only color embedding head (NO upsampling).
    
    Critical for subtitle-only detection:
    - Maintains same resolution as input (H/4, W/4)
    - No ConvTranspose2d → no high-frequency scene text edge restoration
    - Pure subtitle style encoding without scene text noise
    - Enables subtitle binary head to use subtitle style directly
    
    This is essential because:
    - Upsampling in ColorEmbeddingHead restores scene text edges
    - Scene text high-res noise leaks into binary head
    - Subtitle-only detection fails
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 16,
        hidden_ratio: int = 4,
        bias: bool = False,
        normalize: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        hidden_channels = max(in_channels // hidden_ratio, embed_dim)
        
        # NO upsampling - maintains same resolution as input
        # This prevents scene text high-frequency edge restoration
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, embed_dim, kernel_size=3, padding=1, bias=True),
        )
        self.normalize = normalize
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.projection(x)
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1, eps=self.eps)
        return embedding

