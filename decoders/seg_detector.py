from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d


class SubtitleFeatureExtractor(nn.Module):
    """
    Minimal subtitle feature extractor (SFE).
    Projects fuse features (with coordinates) into subtitle-specific feature space
    using a shallow residual stack plus dilated convolutions.
    """

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2, bias=bias),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4, bias=bias),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out


class SubtitleStyleGate(nn.Module):
    """
    Lightweight gating head (SSG) that predicts subtitle probability map (1xHxW).
    """

    def __init__(self, in_channels, bias=False):
        super().__init__()
        hidden = max(in_channels // 2, 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=bias),
            BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class SubtitleBinaryHead(nn.Module):
    """
    Subtitle binary predictor (SBH) that upsamples gated features back to image scale.
    """

    def __init__(self, in_channels, bias=False):
        super().__init__()
        mid = max(in_channels // 4, 16)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=bias),
            BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=2, dilation=2, bias=bias),
            BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid, mid, 2, 2),
            BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid, 1, 2, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

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
        # Legacy params kept for backward compatibility with YAML configs.
        self.enable_color_embedding = enable_color_embedding
        self.color_embed_dim = color_embed_dim
        self.color_normalize = color_normalize

        # Subtitle branch related ***************************
        self.enable_subtitle_branch = kwargs.get('enable_subtitle_branch', False)
        subtitle_inner_channels = kwargs.get('subtitle_inner_channels', inner_channels)
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

        # Subtitle branch ***********************************
        if self.enable_subtitle_branch:
            coord_channels = 1
            self.subtitle_feature_extractor = SubtitleFeatureExtractor(
                inner_channels + coord_channels, subtitle_inner_channels, bias=bias
            )
            self.subtitle_residual_proj = nn.Sequential(
                nn.Conv2d(inner_channels, subtitle_inner_channels, kernel_size=1, bias=bias),
                BatchNorm2d(subtitle_inner_channels)
            )
            self.subtitle_style_gate = SubtitleStyleGate(subtitle_inner_channels, bias=bias)
            self.subtitle_binary_head = SubtitleBinaryHead(subtitle_inner_channels, bias=bias)

            self.subtitle_feature_extractor.apply(self.weights_init)
            self.subtitle_residual_proj.apply(self.weights_init)
            self.subtitle_style_gate.apply(self.weights_init)
            self.subtitle_binary_head.apply(self.weights_init)
        # ***************************************************
    
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
        
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)                                    # 결합된 특징 (fuse)를 입력으로 p-map(biinary) 생성
        
        # Subtitle branch forward **************************************
        subtitle_binary = None
        subtitle_feature = None
        style_gate = None
        if self.enable_subtitle_branch:
            # -----------------------------------------------------
            # 위치 정보를 포함한 fuse 생성
            # -----------------------------------------------------
            N, C, H, W = fuse.shape
            device = fuse.device
            dtype = fuse.dtype

            # y: [0, 1] (top → bottom)
            y_range = torch.linspace(0.0, 2.0, H, device=device, dtype=dtype)
            y_map = y_range.view(1, 1, H, 1).expand(N, 1, H, W)

            fuse_with_coord = torch.cat([fuse, y_map], dim=1)
            subtitle_feature = self.subtitle_feature_extractor(fuse_with_coord)          # fuse → subtitle 전용 feature (1/4 resolution)
            residual_feature = self.subtitle_residual_proj(fuse)
            subtitle_feature = subtitle_feature + residual_feature
            style_gate = self.subtitle_style_gate(subtitle_feature)  # (N, 1, H/4, W/4)
            soft_gate = style_gate.pow(2) + 0.3  # soft attenuation to preserve baseline stroke detail
            gated_subtitle_feature = subtitle_feature * soft_gate
            subtitle_binary = self.subtitle_binary_head(gated_subtitle_feature)  # subtitle binary map (4x upsampled to full resolution)
        # **************************************************************
        
        # Inference mode: return appropriate output
        if not self.training:
            # Return dict to allow representer to select which prediction to use
            result = OrderedDict(binary=binary)
            if self.enable_subtitle_branch and subtitle_binary is not None:
                result.update(subtitle_binary=subtitle_binary)
                if subtitle_feature is not None:
                    result.update(subtitle_feature=subtitle_feature,
                                  subtitle_gate=style_gate)
            return result
        
        # Training mode: return dict with all outputs
        result = OrderedDict(binary=binary)                             # 이후에 thresh, thresh_binary를 추가하기 위해 딕셔너리 형태로 만듦. loss 계산 시 pred가 dict 형태여야 함 (e.g., {'binary': binary, 'thresh': thresh, 'thresh_binary': thresh_binary})
        if self.enable_subtitle_branch:
            result.update(subtitle_binary=subtitle_binary)
            if subtitle_feature is not None:
                result.update(subtitle_feature=subtitle_feature)
            if style_gate is not None:
                result.update(subtitle_gate=style_gate)
        
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
