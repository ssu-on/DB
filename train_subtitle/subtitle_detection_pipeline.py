#!/usr/bin/env python3
"""
Subtitle Detection Pipeline
Domain-specific selective supervision + adaptive inference refinement

이 파이프라인은 다음과 같은 특징을 가집니다:
1. Training: subtitle 영역에서만 loss 계산 (selective supervision)
2. Inference: DBNet + visual heuristic refinement
3. Real-time: 효율적인 subtitle detection
"""

import sys
import os
# 상위 디렉토리를 Python path에 추가 (train_subtitle 폴더에서 상위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class SubtitleRegionDetector:
    """
    자막 영역을 미리 감지하여 training/inference에서 활용
    """
    
    def __init__(self, 
                 subtitle_region_ratio: float = 0.3,  # 하단 30% 영역
                 brightness_threshold: float = 0.8,   # 밝기 임계값
                 contrast_threshold: float = 0.3):    # 대비 임계값
        self.subtitle_region_ratio = subtitle_region_ratio
        self.brightness_threshold = brightness_threshold
        self.contrast_threshold = contrast_threshold
    
    def detect_subtitle_region(self, image: np.ndarray) -> np.ndarray:
        """
        이미지에서 자막이 있을 가능성이 높은 영역을 감지
        
        Args:
            image: 입력 이미지 (H, W, 3)
            
        Returns:
            subtitle_region_mask: 자막 영역 마스크 (H, W)
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. 하단 영역만 고려
        bottom_start = int(h * (1 - self.subtitle_region_ratio))
        bottom_region = image[bottom_start:, :, :]
        
        # 2. 밝기 기반 필터링 (흰색 계열 자막)
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        bright_mask = gray > (self.brightness_threshold * 255)
        
        # 3. 대비 기반 필터링
        contrast = cv2.Laplacian(gray, cv2.CV_64F)
        contrast_mask = np.abs(contrast) > (self.contrast_threshold * 255)
        
        # 4. 결합
        combined_mask = bright_mask & contrast_mask
        
        # 5. Morphological operations으로 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), 
                                       cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 6. 마스크를 전체 이미지 크기로 확장
        mask[bottom_start:, :] = combined_mask
        
        return mask


class SubtitleAwareLoss(nn.Module):
    """
    자막 영역에서만 loss를 계산하는 selective supervision loss
    """
    
    def __init__(self, base_loss_fn, subtitle_detector: SubtitleRegionDetector):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.subtitle_detector = subtitle_detector
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor, 
                image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 모델 예측 결과 (N, 1, H, W)
            gt: Ground truth (N, 1, H, W)
            image: 원본 이미지 (N, 3, H, W)
        """
        # 기본 loss 계산
        base_loss = self.base_loss_fn(pred, gt)
        
        # 자막 영역 마스크 생성
        batch_size = pred.size(0)
        subtitle_masks = []
        
        for i in range(batch_size):
            # 텐서를 numpy로 변환
            img_np = image[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # 자막 영역 감지
            subtitle_mask = self.subtitle_detector.detect_subtitle_region(img_np)
            subtitle_masks.append(torch.from_numpy(subtitle_mask).float())
        
        # 마스크를 텐서로 변환
        subtitle_mask_tensor = torch.stack(subtitle_masks).to(pred.device)
        subtitle_mask_tensor = subtitle_mask_tensor.unsqueeze(1)  # (N, 1, H, W)
        
        # 자막 영역에서만 loss 계산
        masked_loss = base_loss * subtitle_mask_tensor
        
        # 정규화 (자막 영역 픽셀 수로 나누기)
        valid_pixels = subtitle_mask_tensor.sum()
        if valid_pixels > 0:
            final_loss = masked_loss.sum() / valid_pixels
        else:
            final_loss = base_loss.mean()
        
        return final_loss


class SubtitleRefinementModule:
    """
    DBNet 결과를 자막 특화로 정제하는 모듈
    """
    
    def __init__(self, 
                 subtitle_detector: SubtitleRegionDetector,
                 color_threshold: float = 0.8,
                 use_morphology: bool = True):
        self.subtitle_detector = subtitle_detector
        self.color_threshold = color_threshold
        self.use_morphology = use_morphology
    
    def refine_segmentation(self, 
                           binary_map: np.ndarray, 
                           original_image: np.ndarray) -> np.ndarray:
        """
        DBNet의 binary map을 자막 특화로 정제
        
        Args:
            binary_map: DBNet 출력 (H, W)
            original_image: 원본 이미지 (H, W, 3)
            
        Returns:
            refined_mask: 정제된 자막 마스크 (H, W)
        """
        h, w = binary_map.shape
        
        # 1. 자막 영역 마스크 생성
        subtitle_region_mask = self.subtitle_detector.detect_subtitle_region(original_image)
        
        # 2. 자막 영역에서만 binary map 유지
        refined_mask = binary_map & subtitle_region_mask
        
        # 3. 색상 기반 추가 필터링
        if self.color_threshold > 0:
            # 하단 영역에서 밝은 색상만 유지
            bottom_start = int(h * (1 - self.subtitle_detector.subtitle_region_ratio))
            bottom_region = original_image[bottom_start:, :, :]
            
            # RGB 각 채널이 모두 임계값 이상인 픽셀만 유지
            bright_pixels = np.all(bottom_region > (self.color_threshold * 255), axis=2)
            
            # 하단 영역에만 적용
            refined_mask[bottom_start:, :] = (
                refined_mask[bottom_start:, :] & bright_pixels
            )
        
        # 4. Morphological operations
        if self.use_morphology:
            kernel = np.ones((3, 3), np.uint8)
            refined_mask = cv2.morphologyEx(refined_mask.astype(np.uint8), 
                                          cv2.MORPH_CLOSE, kernel)
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        
        return refined_mask.astype(np.uint8)


class SubtitleDetectionPipeline:
    """
    전체 subtitle detection 파이프라인
    """
    
    def __init__(self, 
                 model,  # DBNet 모델
                 subtitle_detector: SubtitleRegionDetector,
                 refinement_module: SubtitleRefinementModule,
                 threshold: float = 0.3):
        self.model = model
        self.subtitle_detector = subtitle_detector
        self.refinement_module = refinement_module
        self.threshold = threshold
    
    def inference(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        자막 detection inference
        
        Args:
            image: 입력 이미지 (H, W, 3)
            
        Returns:
            results: detection 결과 딕셔너리
        """
        # 1. 이미지 전처리
        h, w = image.shape[:2]
        
        # 2. DBNet inference
        with torch.no_grad():
            # 이미지를 텐서로 변환 (여기서는 간단히 처리)
            # 실제로는 DBNet의 전처리 과정을 따라야 함
            pred = self.model(image)  # 실제로는 적절한 전처리 필요
        
        # 3. Binary map 생성
        if isinstance(pred, torch.Tensor):
            binary_map = (pred.cpu().numpy()[0, 0] > self.threshold).astype(np.uint8)
        else:
            binary_map = (pred['binary'][0, 0].cpu().numpy() > self.threshold).astype(np.uint8)
        
        # 4. 자막 특화 정제
        refined_mask = self.refinement_module.refine_segmentation(binary_map, image)
        
        # 5. 결과 반환
        results = {
            'raw_binary_map': binary_map,
            'subtitle_region_mask': self.subtitle_detector.detect_subtitle_region(image),
            'refined_mask': refined_mask,
            'original_image': image
        }
        
        return results
    
    def save_results(self, results: Dict[str, np.ndarray], 
                    output_dir: str, filename: str):
        """
        결과를 파일로 저장
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 원시 binary map
        cv2.imwrite(os.path.join(output_dir, f'{filename}_raw_binary.png'), 
                   results['raw_binary_map'] * 255)
        
        # 자막 영역 마스크
        cv2.imwrite(os.path.join(output_dir, f'{filename}_subtitle_region.png'), 
                   results['subtitle_region_mask'] * 255)
        
        # 정제된 마스크
        cv2.imwrite(os.path.join(output_dir, f'{filename}_refined.png'), 
                   results['refined_mask'] * 255)
        
        # 오버레이 이미지
        overlay = results['original_image'].copy()
        overlay[results['refined_mask'] > 0] = [0, 255, 0]  # 초록색으로 표시
        cv2.imwrite(os.path.join(output_dir, f'{filename}_overlay.jpg'), overlay)


def create_subtitle_detection_pipeline(model, 
                                     subtitle_region_ratio: float = 0.3,
                                     brightness_threshold: float = 0.8,
                                     color_threshold: float = 0.8,
                                     detection_threshold: float = 0.3):
    """
    Subtitle detection 파이프라인 생성 함수
    """
    # 1. 자막 영역 감지기 생성
    subtitle_detector = SubtitleRegionDetector(
        subtitle_region_ratio=subtitle_region_ratio,
        brightness_threshold=brightness_threshold
    )
    
    # 2. 정제 모듈 생성
    refinement_module = SubtitleRefinementModule(
        subtitle_detector=subtitle_detector,
        color_threshold=color_threshold
    )
    
    # 3. 전체 파이프라인 생성
    pipeline = SubtitleDetectionPipeline(
        model=model,
        subtitle_detector=subtitle_detector,
        refinement_module=refinement_module,
        threshold=detection_threshold
    )
    
    return pipeline


# 사용 예시
if __name__ == "__main__":
    print("Subtitle Detection Pipeline")
    print("=" * 50)
    print("1. Training: Selective supervision in subtitle regions")
    print("2. Inference: DBNet + Visual heuristic refinement")
    print("3. Real-time: Efficient subtitle detection")
    print("=" * 50)
    
    # 파라미터 설명
    print("\n주요 파라미터:")
    print("- subtitle_region_ratio: 하단 영역 비율 (기본값: 0.3)")
    print("- brightness_threshold: 밝기 임계값 (기본값: 0.8)")
    print("- color_threshold: 색상 임계값 (기본값: 0.8)")
    print("- detection_threshold: DBNet 임계값 (기본값: 0.3)")
