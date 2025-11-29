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
        return self.decoder(self.backbone(data), *args, **kwargs)


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

        # Keep the raw args for later (e.g., deciding trainable params)
        self.args = args
        # Whether we are in Stage 2 subtitle-only training mode.
        # In this mode, optimizer should only update subtitle branch parameters.
        self.stage2_subtitle_only = args.get('stage2_subtitle_only', False)

        self.model = BasicModel(args)
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
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()
        pred = self.model(data, training=self.training)

        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred

    # ---- Stage 2: subtitle-only optimizer parameter groups ----
    # Trainer will use this, if available, instead of model.parameters().
    def get_trainable_parameters(self):
        """
        Return parameters to be optimized.

        - Default: all parameters (Stage 1 or generic training)
        - Stage 2 (subtitle-only): only subtitle branch parameters
          on top of F5, so that backbone / DBNet heads stay frozen.
        """
        if not self.stage2_subtitle_only:
            return self.parameters()

        # Model is wrapped by DataParallel / DistributedDataParallel
        backbone_decoder = self.model.module
        decoder = backbone_decoder.decoder

        # If subtitle branch is not enabled in decoder, fall back safely.
        if not getattr(decoder, 'subtitle_branch', False):
            return self.parameters()

        params = []
        if hasattr(decoder, 'subtitle_adapter'):
            params.extend(list(decoder.subtitle_adapter.parameters()))
        if hasattr(decoder, 'subtitle_s_head'):
            params.extend(list(decoder.subtitle_s_head.parameters()))
        if hasattr(decoder, 'subtitle_e_head'):
            params.extend(list(decoder.subtitle_e_head.parameters()))

        # Fallback: if nothing was collected, do not crash.
        if not params:
            return self.parameters()

        return params