#!/usr/bin/env python3
"""
Subtitle Detection 테스트 스크립트
DBNet + Subtitle Detection 통합 테스트
"""

import os
import sys
import argparse
# 상위 디렉토리를 Python path에 추가 (train_subtitle 폴더에서 상위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config


def main():
    parser = argparse.ArgumentParser(description='Subtitle Detection 테스트')
    parser.add_argument('--exp', type=str, required=True,
                        help='실험 설정 파일 경로')
    parser.add_argument('--resume', type=str, required=True,
                        help='모델 체크포인트 경로')
    parser.add_argument('--image_path', type=str, required=True,
                        help='테스트할 이미지 경로')
    parser.add_argument('--output_dir', type=str, default='./subtitle_test_results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--subtitle_region_ratio', type=float, default=0.3,
                        help='자막 영역 비율 (하단)')
    parser.add_argument('--brightness_threshold', type=float, default=0.8,
                        help='밝기 임계값')
    parser.add_argument('--color_threshold', type=float, default=0.8,
                        help='색상 임계값')
    parser.add_argument('--no_refinement', action='store_true',
                        help='정제 과정 생략')
    
    args = parser.parse_args()
    
    print("🎬 Subtitle Detection 테스트 시작")
    print("=" * 50)
    print(f"실험 설정: {args.exp}")
    print(f"모델 경로: {args.resume}")
    print(f"테스트 이미지: {args.image_path}")
    print(f"결과 저장 위치: {args.output_dir}")
    print(f"자막 영역 비율: {args.subtitle_region_ratio}")
    print(f"밝기 임계값: {args.brightness_threshold}")
    print(f"색상 임계값: {args.color_threshold}")
    print(f"정제 사용: {not args.no_refinement}")
    print("=" * 50)
    
    try:
        # 1. 실험 설정 로드
        print("📋 실험 설정 로드 중...")
        conf = Config()
        experiment_args = conf.compile(conf.load(args.exp))['Experiment']
        experiment = Configurable.construct_class_from_config(experiment_args)
        print("✅ 실험 설정 로드 완료")
        
        # 2. Geometry-Aware Subtitle Detection 데모 생성
        print("🔧 Geometry-Aware Subtitle Detection 데모 생성 중...")
        from dbnet_geometry_aware_integration import create_subtitle_detection_demo
        
        subtitle_demo = create_subtitle_detection_demo(
            experiment=experiment,
            subtitle_region_ratio=args.subtitle_region_ratio,
            brightness_threshold=args.brightness_threshold,
            color_threshold=args.color_threshold,
            use_refinement=not args.no_refinement,
            aspect_ratio_threshold=3.0,
            height_min=0.02,
            height_max=0.15,
            bottom_ratio=0.3
        )
        print("✅ Subtitle Detection 데모 생성 완료")
        
        # 3. Inference 실행
        print("🚀 Inference 실행 중...")
        subtitle_demo.inference(args.image_path, args.output_dir)
        print("✅ Inference 완료")
        
        # 4. 결과 확인
        print("\n📊 결과 파일들:")
        if os.path.exists(args.output_dir):
            files = os.listdir(args.output_dir)
            for file in sorted(files):
                if file.endswith(('.png', '.jpg')):
                    print(f"  - {file}")
        
        print("\n🎉 테스트 완료!")
        print(f"결과는 {args.output_dir} 디렉토리에서 확인하세요.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def test_subtitle_region_detection():
    """자막 영역 감지 테스트"""
    print("\n🧪 자막 영역 감지 테스트")
    print("-" * 30)
    
    from dbnet_subtitle_integration import SubtitleRegionDetector
    
    # 테스트 이미지 생성 (가상의 자막이 있는 이미지)
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 50  # 어두운 배경
    
    # 하단에 밝은 자막 영역 추가
    test_image[400:450, 100:500, :] = 255  # 흰색 자막
    test_image[420:430, 120:480, :] = 0    # 검은색 텍스트
    
    # 자막 영역 감지기 생성
    detector = SubtitleRegionDetector(
        subtitle_region_ratio=0.3,
        brightness_threshold=0.8
    )
    
    # 자막 영역 감지
    subtitle_mask = detector.detect_subtitle_region(test_image)
    
    # 결과 시각화
    result_image = test_image.copy()
    result_image[subtitle_mask > 0] = [0, 255, 0]  # 초록색으로 표시
    
    # 결과 저장
    os.makedirs('./test_results', exist_ok=True)
    cv2.imwrite('./test_results/test_subtitle_region.png', subtitle_mask * 255)
    cv2.imwrite('./test_results/test_subtitle_region_vis.jpg', result_image)
    
    print("✅ 자막 영역 감지 테스트 완료")
    print("결과: ./test_results/ 디렉토리에서 확인하세요")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # 인자 없이 실행하면 테스트 모드
        test_subtitle_region_detection()
    else:
        # 인자와 함께 실행하면 메인 함수 실행
        sys.exit(main())
