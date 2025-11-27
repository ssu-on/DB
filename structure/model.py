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

        self.model = BasicModel(args)
        # for loading models / (D)DP wrapping
        self.model = parallelize(self.model, distributed, local_rank)

        # Optional: Stage 2 subtitle fine-tuning freeze strategy
        # - Freeze backbone except top block (conv5 / layer4)
        # - Freeze original DBNet text head
        # - Train only conv5 + FeatureAdapter + SubtitleStyleHead
        if args.get('freeze_backbone_for_stage2', False):
            self._freeze_for_subtitle_stage2()

        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def _get_inner_model(self):
        """Unwrap DataParallel / DDP to access backbone & decoder."""
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self.model.module
        return self.model

    def _freeze_for_subtitle_stage2(self):
        """
        Stage 2 subtitle specialization:
          - Freeze backbone except the top block (layer4, conv5).
          - If decoder is a SubtitleSegDetector, freeze the internal text head (detector),
            but keep FeatureAdapter + SubtitleStyleHead trainable.
        """
        inner = self._get_inner_model()
        backbone = getattr(inner, 'backbone', None)
        decoder = getattr(inner, 'decoder', None)

        # 1) Freeze entire backbone
        if backbone is not None:
            for p in backbone.parameters():
                p.requires_grad = False
            # 2) Re-enable conv5 (layer4) only
            if hasattr(backbone, 'layer4'):
                for p in backbone.layer4.parameters():
                    p.requires_grad = True

        # 3) If decoder is SubtitleSegDetector, freeze only the internal SegDetector
        #    (DBNet text head) and keep subtitle branch trainable.
        if decoder is not None and hasattr(decoder, 'detector'):
            # Freeze original DBNet text head
            for p in decoder.detector.parameters():
                p.requires_grad = False
            # FeatureAdapter + SubtitleStyleHead remain trainable by default

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

