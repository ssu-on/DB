#!/usr/bin/env python3
"""
Geometry-Aware Self-Refined Subtitle Detection Framework
P-map 기반 geometry-aware mask 생성 및 self-refinement 모듈
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import ndimage
from skimage.measure import label, regionprops


class GeometryAwareMaskGenerator(nn.Module):
    """
    P-map을 분석하여 자막의 기하학적 특성을 반영한 mask 생성
    """
    
    def __init__(self,
                 aspect_ratio_threshold: float = 3.0,  # τ_ar
                 height_min: float = 0.02,             # h_min (이미지 높이 대비)
                 height_max: float = 0.15,             # h_max (이미지 높이 대비)
                 bottom_ratio: float = 0.3,            # τ_bottom (하단 비율)
                 min_area: int = 100,                  # 최소 blob 면적
                 confidence_threshold: float = 0.3):   # P-map 임계값
        super().__init__()
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.height_min = height_min
        self.height_max = height_max
        self.bottom_ratio = bottom_ratio
        self.min_area = min_area
        self.confidence_threshold = confidence_threshold
    
    def forward(self, p_map: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
        """
        P-map에서 geometry-aware mask 생성
        
        Args:
            p_map: Probability map (N, 1, H, W)
            image_shape: 원본 이미지 크기 (H, W)
            
        Returns:
            geometry_mask: Geometry-aware mask (N, 1, H, W)
        """
        batch_size = p_map.size(0)
        geometry_masks = []
        
        for i in range(batch_size):
            # P-map을 numpy로 변환
            p_np = p_map[i, 0].cpu().numpy()
            
            # Geometry-aware mask 생성
            geo_mask = self._generate_geometry_mask(p_np, image_shape)
            geometry_masks.append(torch.from_numpy(geo_mask).float())
        
        # 배치로 결합
        geometry_mask = torch.stack(geometry_masks).to(p_map.device)
        return geometry_mask.unsqueeze(1)  # (N, 1, H, W)
    
    def _generate_geometry_mask(self, p_map: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        단일 P-map에서 geometry-aware mask 생성
        """
        h, w = p_map.shape
        img_h, img_w = image_shape
        
        # 1. P-map 이진화
        binary_map = (p_map > self.confidence_threshold).astype(np.uint8)
        
        # 2. Morphological operations으로 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
        
        # 3. Connected components 분석
        labeled_map = label(binary_map)
        regions = regionprops(labeled_map)
        
        # 4. Geometry-aware mask 초기화
        geometry_mask = np.zeros_like(p_map, dtype=np.uint8)
        
        # 5. 각 blob에 대해 기하학적 조건 검사
        for region in regions:
            if self._is_subtitle_like_region(region, img_h, img_w):
                # 조건을 만족하는 영역을 mask에 추가
                coords = region.coords
                geometry_mask[coords[:, 0], coords[:, 1]] = 1
        
        return geometry_mask
    
    def _is_subtitle_like_region(self, region, img_h: int, img_w: int) -> bool:
        """
        blob이 자막 유사 영역인지 판단
        """
        # 1. 면적 조건
        if region.area < self.min_area:
            return False
        
        # 2. Aspect ratio 조건 (w/h > τ_ar)
        bbox = region.bbox
        height = bbox[2] - bbox[0]  # max_row - min_row
        width = bbox[3] - bbox[1]  # max_col - min_col
        
        if height == 0:
            return False
        
        aspect_ratio = width / height
        if aspect_ratio < self.aspect_ratio_threshold:
            return False
        
        # 3. 높이 조건 (h_min <= height/img_h <= h_max)
        relative_height = height / img_h
        if not (self.height_min <= relative_height <= self.height_max):
            return False
        
        # 4. 위치 조건 (하단 τ_bottom 비율 이상)
        center_y = (bbox[0] + bbox[2]) / 2  # 중심 y좌표
        relative_y = center_y / img_h
        if relative_y < (1 - self.bottom_ratio):
            return False
        
        return True


class SelfRefinementModule(nn.Module):
    """
    학습 중 예측 맵을 분석하여 subtitle-like 영역을 점진적으로 강화
    """
    
    def __init__(self,
                 geometry_generator: GeometryAwareMaskGenerator,
                 refinement_strength: float = 1.0,
                 adaptive_weighting: bool = True):
        super().__init__()
        self.geometry_generator = geometry_generator
        self.refinement_strength = refinement_strength
        self.adaptive_weighting = adaptive_weighting
        
        # 학습 가능한 가중치 파라미터
        if adaptive_weighting:
            self.geometry_weight = nn.Parameter(torch.tensor(1.0))
            self.confidence_weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, 
                p_map: torch.Tensor, 
                t_map: torch.Tensor,
                binary_map: torch.Tensor,
                image_shape: Tuple[int, int],
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        Self-refinement 수행
        
        Args:
            p_map: Probability map (N, 1, H, W)
            t_map: Threshold map (N, 1, H, W)
            binary_map: Binary map (N, 1, H, W)
            image_shape: 원본 이미지 크기 (H, W)
            epoch: 현재 epoch (점진적 강화용)
            
        Returns:
            refinement_info: 정제 정보 딕셔너리
        """
        # 1. Geometry-aware mask 생성
        geometry_mask = self.geometry_generator(p_map, image_shape)
        
        # 2. 점진적 강화 (epoch에 따라 강도 조절)
        refinement_factor = self._get_refinement_factor(epoch)
        
        # 3. 가중치 계산
        if self.adaptive_weighting:
            weights = self._compute_adaptive_weights(p_map, geometry_mask)
        else:
            weights = geometry_mask
        
        # 4. 정제된 마스크 생성
        refined_mask = self._refine_masks(p_map, t_map, binary_map, weights, refinement_factor)
        
        return {
            'geometry_mask': geometry_mask,
            'refined_weights': weights,
            'refined_mask': refined_mask,
            'refinement_factor': refinement_factor
        }
    
    def _get_refinement_factor(self, epoch: int) -> float:
        """점진적 강화를 위한 factor 계산"""
        # 초기 10 epoch는 약하게, 이후 점진적으로 강화
        if epoch < 10:
            return 0.5 + 0.05 * epoch
        else:
            return min(1.0, 0.5 + 0.1 * (epoch - 10))
    
    def _compute_adaptive_weights(self, 
                                p_map: torch.Tensor, 
                                geometry_mask: torch.Tensor) -> torch.Tensor:
        """적응적 가중치 계산"""
        # Confidence 기반 가중치
        confidence_weight = torch.sigmoid(p_map - 0.5) * 2  # 0~2 범위
        
        # Geometry 기반 가중치
        geometry_weight = geometry_mask.float()
        
        # 결합
        if self.adaptive_weighting:
            combined_weight = (self.confidence_weight * confidence_weight + 
                             self.geometry_weight * geometry_weight)
        else:
            combined_weight = confidence_weight * geometry_weight
        
        return combined_weight
    
    def _refine_masks(self, 
                     p_map: torch.Tensor,
                     t_map: torch.Tensor, 
                     binary_map: torch.Tensor,
                     weights: torch.Tensor,
                     refinement_factor: float) -> torch.Tensor:
        """마스크 정제"""
        # 가중치 적용
        refined_p = p_map * weights * refinement_factor
        refined_t = t_map * weights * refinement_factor
        refined_binary = binary_map * weights * refinement_factor
        
        # 정규화
        refined_p = torch.clamp(refined_p, 0, 1)
        refined_t = torch.clamp(refined_t, 0, 1)
        refined_binary = torch.clamp(refined_binary, 0, 1)
        
        return {
            'refined_p': refined_p,
            'refined_t': refined_t,
            'refined_binary': refined_binary
        }


class GeometryAwareLoss(nn.Module):
    """
    Geometry-aware mask를 적용한 손실 함수
    """
    
    def __init__(self, 
                 base_loss_fn,
                 geometry_generator: GeometryAwareMaskGenerator,
                 self_refinement: SelfRefinementModule,
                 loss_weights: Dict[str, float] = None):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.geometry_generator = geometry_generator
        self.self_refinement = self_refinement
        
        # 손실 가중치 설정
        self.loss_weights = loss_weights or {
            'p_loss': 1.0,
            't_loss': 1.0,
            'b_loss': 1.0
        }
    
    def forward(self, 
                pred: Dict[str, torch.Tensor], 
                batch: Dict[str, torch.Tensor],
                epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Geometry-aware 손실 계산
        
        Args:
            pred: 모델 예측 {'binary': tensor, 'thresh': tensor, ...}
            batch: 배치 데이터 {'image': tensor, 'gt': tensor, 'mask': tensor, ...}
            epoch: 현재 epoch
            
        Returns:
            total_loss: 총 손실
            metrics: 손실 메트릭
        """
        # 이미지 크기 정보
        image = batch['image']
        img_h, img_w = image.shape[2], image.shape[3]
        
        # Self-refinement 수행
        refinement_info = self.self_refinement(
            pred['binary'], pred['thresh'], pred['binary'],
            (img_h, img_w), epoch
        )
        
        # 기본 손실 계산
        base_loss, base_metrics = self.base_loss_fn(pred, batch)
        
        # Geometry-aware 가중치 적용
        geometry_mask = refinement_info['geometry_mask']
        refined_weights = refinement_info['refined_weights']
        
        # 각 손실 항에 가중치 적용
        p_loss = self._compute_weighted_loss(
            pred['binary'], batch['gt'], batch['mask'], 
            refined_weights, 'bce'
        )
        
        t_loss = self._compute_weighted_loss(
            pred['thresh'], batch['thresh_map'], batch['thresh_mask'],
            refined_weights, 'l1'
        )
        
        b_loss = self._compute_weighted_loss(
            pred['binary'], batch['gt'], batch['mask'],
            refined_weights, 'dice'
        )
        
        # 총 손실
        total_loss = (self.loss_weights['p_loss'] * p_loss + 
                     self.loss_weights['t_loss'] * t_loss + 
                     self.loss_weights['b_loss'] * b_loss)
        
        # 메트릭 업데이트
        metrics = base_metrics.copy()
        metrics.update({
            'geometry_p_loss': p_loss,
            'geometry_t_loss': t_loss,
            'geometry_b_loss': b_loss,
            'geometry_mask_ratio': geometry_mask.mean(),
            'refinement_factor': refinement_info['refinement_factor']
        })
        
        return total_loss, metrics
    
    def _compute_weighted_loss(self, 
                             pred: torch.Tensor, 
                             gt: torch.Tensor, 
                             mask: torch.Tensor,
                             weights: torch.Tensor,
                             loss_type: str) -> torch.Tensor:
        """가중치 적용 손실 계산"""
        if loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        elif loss_type == 'l1':
            loss = F.l1_loss(pred, gt, reduction='none')
        elif loss_type == 'dice':
            # Dice loss 계산
            intersection = (pred * gt * mask).sum()
            union = (pred * mask).sum() + (gt * mask).sum() + 1e-6
            loss = 1 - 2.0 * intersection / union
            return loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # 가중치 적용
        weighted_loss = (loss * weights * mask).sum() / (weights * mask).sum().clamp(min=1e-6)
        return weighted_loss


def create_geometry_aware_framework(base_loss_fn,
                                   aspect_ratio_threshold: float = 3.0,
                                   height_min: float = 0.02,
                                   height_max: float = 0.15,
                                   bottom_ratio: float = 0.3,
                                   refinement_strength: float = 1.0,
                                   adaptive_weighting: bool = True):
    """
    Geometry-Aware Self-Refined Framework 생성
    """
    # Geometry generator 생성
    geometry_generator = GeometryAwareMaskGenerator(
        aspect_ratio_threshold=aspect_ratio_threshold,
        height_min=height_min,
        height_max=height_max,
        bottom_ratio=bottom_ratio
    )
    
    # Self-refinement module 생성
    self_refinement = SelfRefinementModule(
        geometry_generator=geometry_generator,
        refinement_strength=refinement_strength,
        adaptive_weighting=adaptive_weighting
    )
    
    # Geometry-aware loss 생성
    geometry_loss = GeometryAwareLoss(
        base_loss_fn=base_loss_fn,
        geometry_generator=geometry_generator,
        self_refinement=self_refinement
    )
    
    return geometry_loss, geometry_generator, self_refinement


# 사용 예시
if __name__ == "__main__":
    print("Geometry-Aware Self-Refined Subtitle Detection Framework")
    print("=" * 60)
    print("주요 구성 요소:")
    print("1. GeometryAwareMaskGenerator: P-map 기반 자막 영역 감지")
    print("2. SelfRefinementModule: 점진적 자막 영역 강화")
    print("3. GeometryAwareLoss: 가중치 적용 손실 함수")
    print("=" * 60)

