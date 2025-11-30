import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import backbones
import decoders


class BasicModel(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)                               # 입력 이미지가 backbone을 통과해 feature 추출


def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return nn.DataParallel(model)

class SegDetectorModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        # Subtitle branch freeze mode: freeze backbone/fuse, train only subtitle branch
        self.freeze_for_subtitle_branch = bool(args.get('freeze_for_subtitle_branch', False))
        
        self.model = BasicModel(args)
        
        if self.freeze_for_subtitle_branch:
            self._freeze_for_subtitle_branch(self.model)
        
        # for loading models
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)           # batch image를 GPU로 이동
        else:
            data = batch.to(self.device)
        data = data.float()                                 # image를 float로 변환
        pred = self.model(data, training=self.training)     # 모델을 통해 예측 수행, self.model은 BasicModle 클래스의 인스턴스

        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)  # batch의 key에 해당하는 value(gt, mask, thresh_map, thresh_mask)를 GPU로 이동
            loss_with_metrics = self.criterion(pred, batch) # pred와 정답을 비교해 loss 계산, self는 SegDetectorModel의 instance, loss 클래스는 self.criterion, self,criterion(pred, batch)를 호출하면, SegDetecotrModel이 들고 있는 loss 모듈의 forward가 실행 됨 SubtitleRefinedL1BalanceCELoss.forward()가 실행
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred

    def _get_basic_model(self):
        """Get the underlying BasicModel, handling DataParallel/DistributedDataParallel wrapper."""
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self.model.module
        return self.model
    
    def _freeze_for_subtitle_branch(self, base_model: BasicModel):
        """
        Freeze backbone and main decoder (fuse, binary head, etc.),
        keep only subtitle branch trainable.
        
        This is for Stage 2 training where:
        - backbone/fuse are frozen (pretrained DBNet)
        - Only subtitle_fuse_branch + subtitle_binary_head + subtitle_color_embed_head are trained
        """
        # Freeze backbone
        for param in base_model.backbone.parameters():
            param.requires_grad = False
        
        decoder = base_model.decoder
        
        # Freeze all decoder components by default
        for param in decoder.parameters():
            param.requires_grad = False
        
        # Unfreeze subtitle branch components (supports both legacy and new modules)
        subtitle_modules = [
            'subtitle_feature_extractor',
            'subtitle_residual_proj',
            'subtitle_style_gate',
            'subtitle_binary_head',
            # Backward compatibility for older subtitle branch structure
            'subtitle_fuse_branch',
            'subtitle_color_embed_head',
        ]
        if hasattr(decoder, 'enable_subtitle_branch') and decoder.enable_subtitle_branch:
            unfrozen = False
            for name in subtitle_modules:
                if hasattr(decoder, name):
                    for param in getattr(decoder, name).parameters():
                        param.requires_grad = True
                    unfrozen = True
            if not unfrozen:
                raise RuntimeError(
                    "Subtitle branch enabled but no subtitle modules were unfrozen. "
                    "Expected modules like subtitle_feature_extractor or subtitle_binary_head."
                )
        else:
            raise RuntimeError(
                "freeze_for_subtitle_branch is enabled but decoder.enable_subtitle_branch is False. "
                "Please set enable_subtitle_branch=True in decoder_args.")
    
    def train(self, mode: bool = True):
        """
        Override train() to maintain freeze state during training.
        When freeze_for_subtitle_branch is enabled, backbone and main decoder stay in eval mode.
        """
        super().train(mode)
        if self.freeze_for_subtitle_branch:
            model = self._get_basic_model()
            # Keep backbone in eval mode
            model.backbone.eval()
            
            decoder = model.decoder
            # Keep main decoder components in eval mode
            subtitle_modules = set([
                'subtitle_feature_extractor',
                'subtitle_residual_proj',
                'subtitle_style_gate',
                'subtitle_binary_head',
                'subtitle_fuse_branch',
                'subtitle_color_embed_head',
            ])
            for name, module in decoder.named_children():
                if name in subtitle_modules:
                    module.train(mode)
                else:
                    module.eval()
        return self