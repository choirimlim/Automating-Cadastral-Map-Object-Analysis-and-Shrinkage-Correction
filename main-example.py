"""
지적원도 객체분석 및 신축량 보정 자동화 통합 모듈
- 전체 파이프라인을 실행하는 메인 스크립트
"""

import os
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import image_preprocessing
from segmentation import multiresolution
from boundary_detection import svm_classifier
from correction import affine_transform

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cadastral_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directories(base_dir="data/results"):
    """필요한 디렉토리 생성"""
    directories = [
        f"{base_dir}/preprocessed",
        f"{base_dir}/segmented",
        f"{base_dir}/boundaries",
        f"{base_dir}/corrected"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"디렉토리 생성: {directory}")

def process_cadastral_map(
    input_path,
    output_dir="data/results",
    gcps_file=None,
    model_path=None,
    preprocess_params=None,
    segment_params=None,
    save_intermediates=True,
    visualize=True
):
    """
    지적원도 처리 파이프라인
    
    Args:
        input_path (str): 입력 이미지 경로
        output_dir (str): 출력 디렉토리
        gcps_file (str, optional): GCP 파일 경로
        model_path (str, optional): SVM 모델 경로
        preprocess_params (dict, optional): 전처리 매개변수
        segment_params (dict, optional): 세그먼테이션 매개변수
        save_intermediates (bool): 중간 결과 저장 여부
        visualize (bool): 결과 시각화 여부
        
    Returns:
        dict: 처리 결과 정보
    """
    # 필요한 디렉토리 생성
    create_directories(output_dir)
    
    # 파일명 추출
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 1. 이미지 전처리
    logger.info("1단계: 이미지 전처리 시작")
    processed_image = image_preprocessing.preprocess(input_path, preprocess_params)
    
    if save_intermediates:
        preprocessed_path = f"{output_dir}/preprocessed/{file_name}_preprocessed.tif"
        image_preprocessing.save_preprocessed_image(processed_image, preprocessed_path)
    
    # 2. 객체 세그먼테이션
    logger.info("2단계: 객체 세그먼테이션 시작")
    if not segment_params:
        segment_params = {
            'scale': 75,
            'shape': 0.3,
            'compactness': 0.5
        }
        
    segmenter = multiresolution.MultiResolutionSegmentation(
        scale=segment_params['scale'],
        shape_weight=segment_params['shape'],
        compactness_weight=segment_params['compactness']
    )
    
    segment_map, segments = segmenter.segment(processed_image)
    
    if save_intermediates:
        segmented_path = f"{output_dir}/segmented/{file_name}_segmented.png"
        segmenter.visualize_segments(segmented_path)
    
    # 3. 경계 추출
    logger.info("3단계: 경계 추출 시작")
    boundary_mask = None
    
    if model_path and os.path.exists(model_path):
        # 저장된 모델 사용
        logger.info(f"저장된 모델 사용: {model_path}")
        classifier = svm_classifier.SVMBoundaryClassifier()
        classifier.load_model(model_path)
        
        predictions, _, _ = classifier.predict(segments, processed_image)
        boundary_mask = classifier.visualize_predictions(
            processed_image, segments, predictions,
            f"{output_dir}/boundaries/{file_name}_boundaries.png" if save_intermediates else None
        )
    else:
        # 간단한 규칙 기반 경계 추출
        logger.info("규칙 기반 경계 추출 사용")
        boundary_mask = segmenter.extract_boundary_mask(threshold=0.7)
        
        if save_intermediates:
            plt.figure(figsize=(10, 8))
            plt.title("추출된 경계선")
            plt.imshow(boundary_mask, cmap='gray')
            plt.axis('off')
            plt.savefig(f"{output_dir}/boundaries/{file_name}_boundaries.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. 신축량 보정
    logger.info("4단계: 신축량 보정 시작")
    corrected_image = None
    
    if gcps_file and os.path.exists(gcps_file):
        # GCP 파일에서 기준점 로드
        gcps = load_gcps_from_file(gcps_file)
        
        if gcps:
            # 아핀 변환 적용
            corrector = affine_transform.AffineCorrection(method='affine')
            src_points = [p[0] for p in gcps]
            dst_points = [p[1] for p in gcps]
            
            corrector.set_control_points(src_points, dst_points)
            corrector.calculate_transform()
            
            # 경계선에 보정 적용
            corrected_image = corrector.apply_correction(boundary_mask)
            
            if save_intermediates or visualize:
                corrected_path = f"{output_dir}/corrected/{file_name}_corrected.tif"
                cv2.imwrite(corrected_path, corrected_image)
                
                if visualize:
                    corrector.visualize_correction(
                        boundary_mask, corrected_image,
                        f"{output_dir}/corrected/{file_name}_comparison.png"
                    )
    else:
        logger.warning("GCP 파일이 제공되지 않았습니다. 신축량 보정을 건너뜁니다.")
    
    logger.info("지적원도 처리 완료")
    
    return {
        'input_path': input_path,
        'preprocessed_image': processed_image,
        'segment_map': segment_map,
        'segments_count': len(segments),
        'boundary_mask': boundary_mask,
        'corrected_image': corrected_image
    }

def load_gcps_from_file(gcps_file):
    """
    GCP 파일에서 기준점 로드
    
    Args:
        gcps_file (str): GCP 파일 경로
        
    Returns:
        List[Tuple]: GCP 목록 [(원본 좌표, 참조 좌표), ...]
    """
    gcps = []
    
    try:
        with open(gcps_file, 'r') as f:
            # 헤더 건너뛰기
            header = f.readline()
            
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    # mapX, mapY, pixelX, pixelY 읽기
                    try:
                        dst_x = float(parts[0])
                        dst_y = float(parts[1])
                        src_x = float(parts[2])
                        src_y = float(parts[3])
                        
                        gcps.append(((src_x, src_y), (dst_x, dst_y)))
                    except ValueError:
                        logger.warning(f"잘못된 GCP 형식: {line}")
        
        logger.info(f"GCP 파일에서 {len(gcps)}개 기준점 로드됨")
        return gcps
        
    except Exception as e:
        logger.error(f"GCP 파일 로드 오류: {e}")
        return []

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="지적원도 객체분석 및 신축량 보정 자동화")
    
    parser.add_argument("--input", "-i", required=True, help="입력 지적원도 이미지 경로")
    parser.add_argument("--output", "-o", default="data/results", help="결과 저장 디렉토리")
    parser.add_argument("--gcps", "-g", help="GCP 파일 경로")
    parser.add_argument("--model", "-m", help="경계 분류 모델 경로")
    parser.add_argument("--no-save", action="store_false", dest="save_intermediates", help="중간 결과 저장 안 함")
    parser.add_argument("--no-vis", action="store_false", dest="visualize", help="결과 시각화 안 함")
    
    args = parser.parse_args()
    
    # 입력 파일 존재 확인
    if not os.path.exists(args.input):
        logger.error(f"입력 파일이 존재하지 않습니다: {args.input}")
        return 1
    
    try:
        # 지적원도 처리 파이프라인 실행
        result = process_cadastral_map(
            args.input,
            output_dir=args.output,
            gcps_file=args.gcps,
            model_path=args.model,
            save_intermediates=args.save_intermediates,
            visualize=args.visualize
        )
        
        return 0
    
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
