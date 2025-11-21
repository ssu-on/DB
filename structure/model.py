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

        # @@ self.model = BasicModel(args) (not freeze)
        #self.model = BasicModel(args) 
        
        self.freeze_except_color_head = bool(args.get('freeze_except_color_head', False))
        base_model = BasicModel(args)
        if self.freeze_except_color_head:
            self._freeze_except_color_head(base_model, require_color_head=args.get('decoder_args', {}).get('enable_color_embedding', False))
        
        # for loading models
        # @@ self.model = parallelize(self.model, distributed, local_rank) (not freeze)
        #self.model = parallelize(self.model, distributed, local_rank)
        self.model = parallelize(base_model, distributed, local_rank)
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

# @@ for backbone freeze
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_except_color_head:
            model = self._get_basic_model()
            model.backbone.eval()
            decoder = model.decoder
            for name, module in decoder.named_children():
                if name == 'color_head' and getattr(decoder, 'enable_color_embedding', False):
                    module.train(mode)
                else:
                    module.eval()
        return self

    def _get_basic_model(self):
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self.model.module
        return self.model

    def _freeze_except_color_head(self, base_model: BasicModel, require_color_head: bool = False):
        for param in base_model.backbone.parameters():
            param.requires_grad = False
        decoder = base_model.decoder
        for param in decoder.parameters():
            param.requires_grad = False
        if getattr(decoder, 'enable_color_embedding', False) and hasattr(decoder, 'color_head'):
            for param in decoder.color_head.parameters():
                param.requires_grad = True
        elif require_color_head:
            raise RuntimeError("freeze_except_color_head is enabled but decoder.color_head is missing.")