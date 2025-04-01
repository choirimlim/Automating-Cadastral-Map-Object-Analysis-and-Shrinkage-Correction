"""
지적원도 이미지 전처리 모듈
- 노이즈 제거, 선명화, 대비 강화 등의 기능 제공
"""

import cv2
import numpy as np
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image(image_path):
    """
    이미지 파일을 로드합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        
    Returns:
        numpy.ndarray: 로드된 이미지
    """
    try:
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
        logger.info(f"이미지 로드 완료: {image_path} (크기: {image.shape})")
        return image
    
    except Exception as e:
        logger.error(f"이미지 로드 실패: {e}")
        raise

def convert_to_grayscale(image):
    """
    컬러 이미지를 그레이스케일로 변환합니다.
    
    Args:
        image (numpy.ndarray): 변환할 이미지
        
    Returns:
        numpy.ndarray: 그레이스케일 이미지
    """
    if len(image.shape) == 2:
        logger.info("이미지가 이미 그레이스케일입니다.")
        return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logger.info(f"그레이스케일 변환 완료: 크기 {gray.shape}")
    return gray

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    가우시안 블러를 적용하여 노이즈를 제거합니다.
    
    Args:
        image (numpy.ndarray): 입력 이미지
        kernel_size (tuple): 가우시안 커널 크기, 기본값 (5, 5)
        sigma (float): 가우시안 커널의 표준 편차, 기본값 0(자동 계산)
        
    Returns:
        numpy.ndarray: 노이즈가 감소된 이미지
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    logger.info(f"가우시안 블러 적용 완료: 커널 크기 {kernel_size}")
    return blurred

def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    적응형 히스토그램 평활화(CLAHE)를 적용하여 대비를 향상시킵니다.
    
    Args:
        image (numpy.ndarray): 입력 그레이스케일 이미지
        clip_limit (float): 대비 제한 파라미터, 기본값 2.0
        tile_grid_size (tuple): 타일의 그리드 크기, 기본값 (8, 8)
        
    Returns:
        numpy.ndarray: 대비가 향상된 이미지
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    logger.info(f"대비 향상 완료: clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")
    return enhanced

def binarize_image(image, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                  threshold_type=cv2.THRESH_BINARY_INV, block_size=11, constant=2):
    """
    적응형 임계값을 적용하여 이미지를 이진화합니다.
    
    Args:
        image (numpy.ndarray): 입력 그레이스케일 이미지
        adaptive_method: 적응형 임계값 방법, 기본값 cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        threshold_type: 임계값 유형, 기본값 cv2.THRESH_BINARY_INV
        block_size (int): 임계값 계산에 사용되는 블록 크기, 기본값 11
        constant (int): 임계값에서 뺄 상수, 기본값 2
        
    Returns:
        numpy.ndarray: 이진화된 이미지
    """
    binary = cv2.adaptiveThreshold(image, 255, adaptive_method, 
                                 threshold_type, block_size, constant)
    logger.info(f"이미지 이진화 완료: block_size={block_size}, constant={constant}")
    return binary

def morphological_operations(image, operation_type='close', kernel_size=3):
    """
    모폴로지 연산을 적용합니다.
    
    Args:
        image (numpy.ndarray): 입력 이진화 이미지
        operation_type (str): 연산 유형 ('open', 'close', 'dilate', 'erode'), 기본값 'close'
        kernel_size (int): 커널 크기, 기본값 3
        
    Returns:
        numpy.ndarray: 모폴로지 연산이 적용된 이미지
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation_type == 'open':
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        operation_name = "열림 연산"
    elif operation_type == 'close':
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        operation_name = "닫힘 연산"
    elif operation_type == 'dilate':
        result = cv2.dilate(image, kernel)
        operation_name = "팽창 연산"
    elif operation_type == 'erode':
        result = cv2.erode(image, kernel)
        operation_name = "침식 연산"
    else:
        logger.warning(f"지원되지 않는 연산 유형: {operation_type}. 원본 이미지를 반환합니다.")
        return image
    
    logger.info(f"모폴로지 {operation_name} 적용 완료: 커널 크기 {kernel_size}x{kernel_size}")
    return result

def preprocess(image_path, params=None):
    """
    지적원도 이미지에 대한 전체 전처리 파이프라인을 실행합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        params (dict, optional): 전처리 매개변수를 포함하는 딕셔너리
                                기본값 None(기본 매개변수 사용)
    
    Returns:
        numpy.ndarray: 전처리된 이미지
    """
    # 기본 매개변수 설정
    if params is None:
        params = {
            'gaussian_kernel': (5, 5),
            'gaussian_sigma': 0,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8),
            'binary_block_size': 11,
            'binary_constant': 2,
            'morph_operation': 'close',
            'morph_kernel_size': 3
        }
    
    logger.info("이미지 전처리 시작")
    
    # 이미지 로드
    image = load_image(image_path)
    
    # 그레이스케일 변환
    gray = convert_to_grayscale(image)
    
    # 가우시안 블러 적용
    blurred = apply_gaussian_blur(gray, 
                                params['gaussian_kernel'],
                                params['gaussian_sigma'])
    
    # 대비 향상
    enhanced = enhance_contrast(blurred,
                              params['clahe_clip_limit'],
                              params['clahe_grid_size'])
    
    # 이미지 이진화
    binary = binarize_image(enhanced,
                          block_size=params['binary_block_size'],
                          constant=params['binary_constant'])
    
    # 모폴로지 연산 적용
    processed = morphological_operations(binary,
                                       params['morph_operation'],
                                       params['morph_kernel_size'])
    
    logger.info("이미지 전처리 완료")
    return processed

def save_preprocessed_image(image, output_path):
    """
    전처리된 이미지를 파일로 저장합니다.
    
    Args:
        image (numpy.ndarray): 저장할 이미지
        output_path (str): 출력 파일 경로
        
    Returns:
        bool: 저장 성공 여부
    """
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = cv2.imwrite(str(output_path), image)
        if result:
            logger.info(f"이미지 저장 완료: {output_path}")
        else:
            logger.error(f"이미지 저장 실패: {output_path}")
        
        return result
    
    except Exception as e:
        logger.error(f"이미지 저장 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    # 테스트용 코드
    import matplotlib.pyplot as plt
    
    # 테스트 이미지 경로
    test_image_path = "data/sample/raw_cadastral_maps/test_map.tif"
    output_path = "data/results/preprocessed/test_map_preprocessed.tif"
    
    # 전처리 실행
    try:
        processed_image = preprocess(test_image_path)
        save_preprocessed_image(processed_image, output_path)
        
        # 결과 시각화
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("원본 이미지")
        plt.imshow(cv2.imread(test_image_path), cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("전처리된 이미지")
        plt.imshow(processed_image, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}")
