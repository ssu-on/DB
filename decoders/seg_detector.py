from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .color_embedding_head import ColorEmbeddingHead, SubtitleColorEmbeddingHead

BatchNorm2d = nn.BatchNorm2d


class DeepSubtitleFuseBranch(nn.Module):
    """
    Deep subtitle fuse branch with residual connection and dilated convolutions.
    
    Transforms general text features (fuse) into subtitle-only feature space.
    Uses 6 conv layers (4 base + 2 dilated) with residual connection for sufficient 
    capacity to suppress scene text and extract subtitle-specific features.
    
    Structure:
    - Conv3x3 → BN → ReLU
    - Conv3x3 → BN → ReLU
    - ResidualBlock (Conv3x3×2)
    - Conv3x3(dilation=2) → BN → ReLU
    - Conv3x3(dilation=4) → BN → ReLU
    """
    
    def __init__(self, in_channels, out_channels, bias=False):
        super(DeepSubtitleFuseBranch, self).__init__()
        
        # First layer: project to subtitle_inner_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)
        self.bn1 = BatchNorm2d(out_channels)
        
        # Second layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias)
        self.bn2 = BatchNorm2d(out_channels)
        
        # Residual block: 2 conv layers
        self.conv3_res = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias)
        self.bn3_res = BatchNorm2d(out_channels)
        self.conv4_res = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias)
        self.bn4_res = BatchNorm2d(out_channels)
        
        # Dilated convolutions for larger receptive field (subtitle style capture)
        self.conv5_dil2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2, bias=bias)
        self.bn5_dil2 = BatchNorm2d(out_channels)
        
        self.conv6_dil4 = nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4, bias=bias)
        self.bn6_dil4 = BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # First layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Residual block
        residual = out
        out = self.conv3_res(out)
        out = self.bn3_res(out)
        out = self.relu(out)
        out = self.conv4_res(out)
        out = self.bn4_res(out)
        out = out + residual  # Residual connection
        out = self.relu(out)
        
        # Dilated convolutions for larger receptive field
        out = self.conv5_dil2(out)
        out = self.bn5_dil2(out)
        out = self.relu(out)
        
        out = self.conv6_dil4(out)
        out = self.bn6_dil4(out)
        out = self.relu(out)
        
        return out

class SegDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 enable_color_embedding=False, color_embed_dim=16,
                 color_normalize=True, *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector, self).__init__()
        self.k = k
        self.serial = serial

        # Color embedding related ***************************
        self.enable_color_embedding = enable_color_embedding
        self.color_embed_dim = color_embed_dim
        self.color_normalize = color_normalize
        # ***************************************************
        
        # Subtitle branch related ***************************
        self.enable_subtitle_branch = kwargs.get('enable_subtitle_branch', False)
        subtitle_inner_channels = kwargs.get('subtitle_inner_channels', inner_channels)
        # Whether to use color embedding in binary head (essential for subtitle-only detection)
        self.use_color_embedding_in_binary = kwargs.get('use_color_embedding_in_binary', True)
        # Whether to use deep subtitle_fuse_branch (5-layer + residual) for better scene text suppression
        self.use_deep_subtitle_fuse = kwargs.get('use_deep_subtitle_fuse', True)
        # ***************************************************

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

        # Color embedding related ***************************
        if self.enable_color_embedding:
            self.color_head = ColorEmbeddingHead(
                inner_channels,
                embed_dim=self.color_embed_dim,
                bias=bias,
                normalize=self.color_normalize)
            self.color_head.apply(self.weights_init)
        # ***************************************************
        
        # Subtitle branch ***********************************
        if self.enable_subtitle_branch:
            # subtitle_fuse_branch: fuse feature를 subtitle 전용 feature로 변환
            # Deep version (5-layer + residual) for better scene text suppression
            if self.use_deep_subtitle_fuse:
                # Deep branch with residual connection for stronger subtitle-only feature learning
                self.subtitle_fuse_branch = self._build_deep_subtitle_fuse_branch(
                    inner_channels, subtitle_inner_channels, bias)
            else:
                # Shallow branch (original, 2 conv layers)
                self.subtitle_fuse_branch = nn.Sequential(
                    nn.Conv2d(inner_channels, subtitle_inner_channels, 3, padding=1, bias=bias),
                    BatchNorm2d(subtitle_inner_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(subtitle_inner_channels, subtitle_inner_channels, 3, padding=1, bias=bias),
                    BatchNorm2d(subtitle_inner_channels),
                    nn.ReLU(inplace=True))
            
            # subtitle_binary_head: subtitle binary map 생성
            # Input: subtitle_feature + subtitle_color_embedding (concat)
            # This allows binary head to directly use subtitle style information
            binary_input_channels = subtitle_inner_channels
            if self.use_color_embedding_in_binary:
                embed_dim = self.color_embed_dim if self.enable_color_embedding else 16
                binary_input_channels = subtitle_inner_channels + embed_dim
            
            # subtitle_binary_head with dilated convolutions for larger receptive field
            # This enables subtitle style (stroke, blur, color halo, position) detection
            # Essential for distinguishing subtitle from scene text
            self.subtitle_binary_head = nn.Sequential(
                # Base conv
                nn.Conv2d(binary_input_channels, subtitle_inner_channels //
                          4, 3, padding=1, bias=bias),
                BatchNorm2d(subtitle_inner_channels//4),
                nn.ReLU(inplace=True),
                # Dilated conv for larger receptive field (subtitle style capture)
                nn.Conv2d(subtitle_inner_channels//4, subtitle_inner_channels//4, 
                          3, padding=2, dilation=2, bias=bias),
                BatchNorm2d(subtitle_inner_channels//4),
                nn.ReLU(inplace=True),
                # Another dilated conv
                nn.Conv2d(subtitle_inner_channels//4, subtitle_inner_channels//4, 
                          3, padding=4, dilation=4, bias=bias),
                BatchNorm2d(subtitle_inner_channels//4),
                nn.ReLU(inplace=True),
                # Upsampling layers
                nn.ConvTranspose2d(subtitle_inner_channels//4, subtitle_inner_channels//4, 2, 2),
                BatchNorm2d(subtitle_inner_channels//4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(subtitle_inner_channels//4, 1, 2, 2),
                nn.Sigmoid())
            
            # subtitle_color_embed_head: subtitle color embedding 생성
            # Use SubtitleColorEmbeddingHead (NO upsampling) to prevent scene text edge restoration
            # This is critical for subtitle-only detection
            self.subtitle_color_embed_head = SubtitleColorEmbeddingHead(
                subtitle_inner_channels,
                embed_dim=self.color_embed_dim if self.enable_color_embedding else 16,
                bias=bias,
                normalize=self.color_normalize)
            
            self.subtitle_fuse_branch.apply(self.weights_init)
            self.subtitle_binary_head.apply(self.weights_init)
            self.subtitle_color_embed_head.apply(self.weights_init)
        # ***************************************************

    def _build_deep_subtitle_fuse_branch(self, in_channels, out_channels, bias):
        """
        Build deep subtitle fuse branch (5-layer + residual) for better scene text suppression.
        
        This provides sufficient capacity to transform general text features (fuse)
        into subtitle-only feature space, removing scene text information.
        """
        return DeepSubtitleFuseBranch(in_channels, out_channels, bias)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)                           # 다중 스케일 feauture map
        
        color_embedding = None                                          # @@ 각 픽셀마다 추정된 embed_dim의 embedding vector가 들어 있는 tensor. None은 초기값.
        if self.enable_color_embedding:                                 # @@
            color_embedding = self.color_head(fuse)                     # @@ fuse를 입력으로 각 픽셀마다 추정된 embed_dim의 embedding vector가 들어 있는 tensor
            
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)                                    # 결합된 특징 (fuse)를 입력으로 p-map(biinary) 생성
        
        # Subtitle branch forward **************************************
        subtitle_binary = None
        subtitle_color_embedding = None
        if self.enable_subtitle_branch:
            subtitle_feature = self.subtitle_fuse_branch(fuse)          # fuse → subtitle 전용 feature (1/4 resolution)
            # SubtitleColorEmbeddingHead: NO upsampling, maintains same resolution (H/4, W/4)
            # This prevents scene text high-frequency edge restoration
            subtitle_color_embedding = self.subtitle_color_embed_head(subtitle_feature)  # (N, embed_dim, H/4, W/4)
            
            # Concat subtitle_feature + subtitle_color_embedding for binary head
            # This allows binary head to directly use subtitle style information (essential for subtitle-only detection)
            # Both are same resolution (H/4, W/4) - no downsampling needed
            if self.use_color_embedding_in_binary:
                binary_input = torch.cat([subtitle_feature, subtitle_color_embedding], dim=1)  # (N, C+embed_dim, H/4, W/4)
            else:
                # Fallback: use subtitle_feature only (old behavior)
                binary_input = subtitle_feature
            
            subtitle_binary = self.subtitle_binary_head(binary_input)  # subtitle binary map (4x upsampled to full resolution)
        # **************************************************************

        # color_embedding = None                                          # @@ 각 픽셀마다 추정된 embed_dim의 embedding vector가 들어 있는 tensor. None은 초기값.
        # if self.enable_color_embedding:                                 # @@
        #     color_embedding = F.interpolate(
        #         fuse,
        #         size=binary.shape[-2:],
        #         mode='bilinear',
        #         align_corners=False)
        #     if self.color_normalize:
        #         color_embedding = F.normalize(color_embedding, p=2, dim=1, eps=1e-6)

        # if self.training:                                             # train 모드에서는 binary, color_embedding 모두 반환
        #     result = OrderedDict(binary=binary)                       # 이후에 thresh, thresh_binary를 추가하기 위해 딕셔너리 형태로 만듦. loss 계산 시 pred가 dict 형태여야 함 (e.g., {'binary': binary, 'thresh': thresh, 'thresh_binary': thresh_binary})
        # else:
        #     return binary                                             # eval 모드에서는 p-map(binary)만 반환     
        
        # Inference mode: return appropriate output
        if not self.training:
            # Return dict to allow representer to select which prediction to use
            result = OrderedDict(binary=binary)
            if self.enable_subtitle_branch and subtitle_binary is not None:
                result.update(subtitle_binary=subtitle_binary)
            if self.enable_color_embedding and color_embedding is not None:
                result.update(color_embedding=color_embedding)
            return result
        
        # Training mode: return dict with all outputs
        result = OrderedDict(binary=binary)                             # 이후에 thresh, thresh_binary를 추가하기 위해 딕셔너리 형태로 만듦. loss 계산 시 pred가 dict 형태여야 함 (e.g., {'binary': binary, 'thresh': thresh, 'thresh_binary': thresh_binary})
        if self.enable_color_embedding:
            result.update(color_embedding=color_embedding)
        if self.enable_subtitle_branch:
            result.update(subtitle_binary=subtitle_binary)
            result.update(subtitle_color_embedding=subtitle_color_embedding)
        
        if self.adaptive and self.training:                             # adaptive 모드, thresh, binary-thresh까지 추가 반환, DBNet의 핵심 개념으로, 픽셀별로 다른 threshold를 학습
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)                                  # 픽셀별로 서로 다른 threshold를 학습
            thresh_binary = self.step_function(binary, thresh)          # 픽셀별로 binary와 해당 위치의 thresh를 이용해 thresh_binary 생성
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
