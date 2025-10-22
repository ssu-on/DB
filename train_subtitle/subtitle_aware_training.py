#!/usr/bin/env python3
"""
Subtitle-aware Training for DBNet
Selective supervision in subtitle regions
"""

import sys
import os
# 상위 디렉토리를 Python path에 추가 (train_subtitle 폴더에서 상위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class SubtitleAwareLoss(nn.Module):
    """
    자막 영역에서만 loss를 계산하는 selective supervision loss
    """
    
    def __init__(self, 
                 base_loss_fn,
                 subtitle_region_ratio: float = 0.3,
                 brightness_threshold: float = 0.8,
                 contrast_threshold: float = 0.3,
                 loss_weight: float = 1.0):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.subtitle_region_ratio = subtitle_region_ratio
        self.brightness_threshold = brightness_threshold
        self.contrast_threshold = contrast_threshold
        self.loss_weight = loss_weight
    
    def detect_subtitle_region(self, image: torch.Tensor) -> torch.Tensor:
        """
        텐서 형태의 이미지에서 자막 영역 감지
        
        Args:
            image: 입력 이미지 (N, 3, H, W)
            
        Returns:
            subtitle_mask: 자막 영역 마스크 (N, 1, H, W)
        """
        batch_size, channels, height, width = image.shape
        subtitle_masks = []
        
        for i in range(batch_size):
            # 텐서를 numpy로 변환
            img_np = image[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # 자막 영역 감지
            subtitle_mask = self._detect_subtitle_region_np(img_np)
            subtitle_masks.append(torch.from_numpy(subtitle_mask).float())
        
        # 마스크를 텐서로 변환
        subtitle_mask_tensor = torch.stack(subtitle_masks).to(image.device)
        subtitle_mask_tensor = subtitle_mask_tensor.unsqueeze(1)  # (N, 1, H, W)
        
        return subtitle_mask_tensor
    
    def _detect_subtitle_region_np(self, image: np.ndarray) -> np.ndarray:
        """numpy 이미지에서 자막 영역 감지"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 하단 영역만 고려
        bottom_start = int(h * (1 - self.subtitle_region_ratio))
        bottom_region = image[bottom_start:, :, :]
        
        # 밝기 기반 필터링
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        bright_mask = gray > (self.brightness_threshold * 255)
        
        # 대비 기반 필터링
        contrast = cv2.Laplacian(gray, cv2.CV_64F)
        contrast_mask = np.abs(contrast) > (self.contrast_threshold * 255)
        
        # 결합
        combined_mask = bright_mask & contrast_mask
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), 
                                       cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 전체 이미지 크기로 확장
        mask[bottom_start:, :] = combined_mask
        
        return mask
    
    def forward(self, pred: Dict[str, torch.Tensor], 
                batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Subtitle-aware loss 계산
        
        Args:
            pred: 모델 예측 결과 {'binary': tensor, ...}
            batch: 배치 데이터 {'image': tensor, 'gt': tensor, 'mask': tensor, ...}
            
        Returns:
            loss: 계산된 loss
            metrics: loss 메트릭
        """
        # 원본 이미지에서 자막 영역 감지
        image = batch['image']
        subtitle_mask = self.detect_subtitle_region(image)
        
        # 기본 loss 계산
        base_loss, base_metrics = self.base_loss_fn(pred, batch)
        
        # 자막 영역에서만 loss 계산
        if 'binary' in pred:
            binary_pred = pred['binary']
            binary_gt = batch['gt']
            binary_mask = batch['mask']
            
            # 자막 영역 마스크와 기존 마스크 결합
            combined_mask = binary_mask * subtitle_mask
            
            # 자막 영역에서만 loss 계산
            if combined_mask.sum() > 0:
                # Binary loss 계산
                binary_loss = F.binary_cross_entropy_with_logits(
                    binary_pred, binary_gt, reduction='none'
                )
                
                # 자막 영역에서만 loss 적용
                masked_binary_loss = (binary_loss * combined_mask).sum() / combined_mask.sum()
                
                # 최종 loss
                final_loss = self.loss_weight * masked_binary_loss
                
                # 메트릭 업데이트
                metrics = base_metrics.copy()
                metrics['subtitle_binary_loss'] = masked_binary_loss
                metrics['subtitle_region_ratio'] = subtitle_mask.mean()
                
            else:
                # 자막 영역이 없는 경우 기본 loss 사용
                final_loss = base_loss
                metrics = base_metrics
                metrics['subtitle_region_ratio'] = 0.0
        else:
            final_loss = base_loss
            metrics = base_metrics
        
        return final_loss, metrics


class SubtitleAwareDataLoader:
    """
    Subtitle-aware 데이터 로더
    기존 데이터 로더에 subtitle region 정보 추가
    """
    
    def __init__(self, 
                 base_data_loader,
                 subtitle_region_ratio: float = 0.3,
                 brightness_threshold: float = 0.8):
        self.base_data_loader = base_data_loader
        self.subtitle_region_ratio = subtitle_region_ratio
        self.brightness_threshold = brightness_threshold
    
    def __iter__(self):
        for batch in self.base_data_loader:
            # 자막 영역 마스크 추가
            batch = self._add_subtitle_region_info(batch)
            yield batch
    
    def __len__(self):
        return len(self.base_data_loader)
    
    def _add_subtitle_region_info(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """배치에 자막 영역 정보 추가"""
        if 'image' in batch:
            image = batch['image']
            batch_size = image.size(0)
            
            # 자막 영역 마스크 생성
            subtitle_masks = []
            for i in range(batch_size):
                img_np = image[i].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                
                subtitle_mask = self._detect_subtitle_region(img_np)
                subtitle_masks.append(torch.from_numpy(subtitle_mask).float())
            
            # 마스크를 텐서로 변환
            subtitle_mask_tensor = torch.stack(subtitle_masks).to(image.device)
            batch['subtitle_region_mask'] = subtitle_mask_tensor.unsqueeze(1)
        
        return batch
    
    def _detect_subtitle_region(self, image: np.ndarray) -> np.ndarray:
        """이미지에서 자막 영역 감지"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 하단 영역만 고려
        bottom_start = int(h * (1 - self.subtitle_region_ratio))
        bottom_region = image[bottom_start:, :, :]
        
        # 밝기 기반 필터링
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        bright_mask = gray > (self.brightness_threshold * 255)
        
        # 대비 기반 필터링
        contrast = cv2.Laplacian(gray, cv2.CV_64F)
        contrast_mask = np.abs(contrast) > 0.3 * 255
        
        # 결합
        combined_mask = bright_mask & contrast_mask
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), 
                                       cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 전체 이미지 크기로 확장
        mask[bottom_start:, :] = combined_mask
        
        return mask


def create_subtitle_aware_loss(base_loss_fn, 
                              subtitle_region_ratio: float = 0.3,
                              brightness_threshold: float = 0.8,
                              loss_weight: float = 1.0):
    """
    Subtitle-aware loss 생성
    """
    return SubtitleAwareLoss(
        base_loss_fn=base_loss_fn,
        subtitle_region_ratio=subtitle_region_ratio,
        brightness_threshold=brightness_threshold,
        loss_weight=loss_weight
    )


def create_subtitle_aware_data_loader(base_data_loader,
                                    subtitle_region_ratio: float = 0.3,
                                    brightness_threshold: float = 0.8):
    """
    Subtitle-aware 데이터 로더 생성
    """
    return SubtitleAwareDataLoader(
        base_data_loader=base_data_loader,
        subtitle_region_ratio=subtitle_region_ratio,
        brightness_threshold=brightness_threshold
    )


class SubtitleAwareTrainer:
    """
    Subtitle-aware training을 위한 메인 trainer 클래스
    """
    
    def __init__(self, config, resume_path=None):
        self.config = config
        self.resume_path = resume_path
        
        # 기존 DBNet 구조 사용
        from concern.config import Config
        from experiment import Structure, TrainSettings, ValidationSettings, Experiment
        from trainer import Trainer
        
        # Config 로드
        conf = Config()
        experiment_args = conf.compile(conf.load(config['exp']))['Experiment']
        experiment_args.update(cmd=config)
        self.experiment = Configurable.construct_class_from_config(experiment_args)
        
        # Trainer 초기화 (기존 구조 사용)
        self.trainer = Trainer(self.experiment)
        
        # Subtitle-aware loss 설정
        self.subtitle_region_ratio = config.get('subtitle_region_ratio', 0.3)
        self.brightness_threshold = config.get('brightness_threshold', 0.8)
        
        # Training 설정
        self.start_epoch = 0
        self.best_f1 = 0.0
        
        # 체크포인트 로드
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)
    
    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드"""
        # 기존 trainer의 체크포인트 로드 메서드 사용
        pass
    
    def train(self, num_epochs):
        """전체 학습 과정 - 기존 trainer 사용"""
        print(f'Starting Subtitle-aware Training for {num_epochs} epochs')
        print(f'Subtitle region ratio: {self.subtitle_region_ratio}')
        print(f'Brightness threshold: {self.brightness_threshold}')
        
        # 기존 trainer의 train 메서드 사용
        self.trainer.train()


def main():
    """메인 실행 함수 - 기존 train.py와 동일한 구조"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Subtitle-aware Training')
    parser.add_argument('exp', type=str, help='Experiment config file')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--subtitle_ratio', type=float, default=0.3,
                       help='Subtitle region ratio')
    parser.add_argument('--brightness_thresh', type=float, default=0.8,
                       help='Brightness threshold')
    parser.add_argument('--verbose', action='store_true', help='show verbose info')
    parser.add_argument('--visualize', action='store_true', help='visualize maps in tensorboard')
    parser.add_argument('--validate', action='store_true', dest='validate', help='Validate during training')
    parser.add_argument('--no-validate', action='store_false', dest='validate', help='Validate during training')
    parser.add_argument('--print-config-only', action='store_true', help='print config without actual training')
    parser.add_argument('--debug', action='store_true', dest='debug', help='Run with debug mode')
    parser.add_argument('--no-debug', action='store_false', dest='debug', help='Run without debug mode')
    parser.add_argument('--benchmark', action='store_true', dest='benchmark', help='Open cudnn benchmark mode')
    parser.add_argument('--no-benchmark', action='store_false', dest='benchmark', help='Turn cudnn benchmark mode off')
    parser.add_argument('-d', '--distributed', action='store_true', dest='distributed', help='Use distributed training')
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')
    parser.add_argument('-g', '--num_gpus', dest='num_gpus', default=4, type=int, help='The number of accessible gpus')
    parser.set_defaults(debug=False)
    parser.set_defaults(benchmark=True)
    
    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    
    # Subtitle 설정 추가
    args['subtitle_region_ratio'] = args.get('subtitle_ratio', 0.3)
    args['brightness_threshold'] = args.get('brightness_thresh', 0.8)
    
    if args['distributed']:
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    # 기존 DBNet 구조 사용
    from concern.config import Config
    from concern.config import Configurable
    from experiment import Structure, TrainSettings, ValidationSettings, Experiment
    from trainer import Trainer
    
    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)
    
    if not args['print_config_only']:
        torch.backends.cudnn.benchmark = args['benchmark']
        trainer = Trainer(experiment)
        trainer.train()


# 사용 예시
if __name__ == "__main__":
    main()
