"""
다중해상도 세그먼테이션(MRS) 모듈
- 지적원도의 객체 기반 이미지 분석(OBIA)을 위한 세그먼테이션 기능 제공
- eCognition에서 사용되는 MRS 알고리즘을 파이썬으로 구현
"""

import numpy as np
import cv2
from skimage.segmentation import felzenszwalb, quickshift, slic
from skimage.future import graph
from skimage import exposure, morphology, measure
import matplotlib.pyplot as plt
import logging
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Segment:
    """세그먼트 객체를 표현하는 클래스"""
    id: int
    pixels: List[Tuple[int, int]]  # (row, col) 좌표 목록
    mean_color: float
    area: int
    perimeter: int
    shape_index: float
    compactness: float
    neighbors: List[int] = None
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = []

class MultiResolutionSegmentation:
    """
    다중해상도 세그먼테이션 클래스
    - 스케일, 형태, 컴팩트니스 매개변수를 기반으로 이미지를 세그먼트화
    """
    
    def __init__(self, scale=50, shape_weight=0.3, compactness_weight=0.5):
        """
        초기화
        
        Args:
            scale (int): 세그먼트 크기를 제어하는 스케일 매개변수 (기본값: 50)
            shape_weight (float): 형태의 중요도 (0~1 사이, 기본값: 0.3)
            compactness_weight (float): 컴팩트니스의 중요도 (0~1 사이, 기본값: 0.5)
        """
        self.scale = scale
        self.shape_weight = shape_weight
        self.compactness_weight = compactness_weight
        self.segments = []
        self.segment_map = None
        self.input_image = None
        logger.info(f"다중해상도 세그먼테이션 초기화: scale={scale}, shape_weight={shape_weight}, compactness_weight={compactness_weight}")
    
    def segment(self, image):
        """
        이미지 세그먼테이션 수행
        
        Args:
            image (numpy.ndarray): 입력 이미지 (그레이스케일 또는 이진화 이미지)
            
        Returns:
            numpy.ndarray: 세그먼트 레이블이 포함된 맵
            List[Segment]: 세그먼트 객체 목록
        """
        self.input_image = image.copy()
        height, width = image.shape[:2]
        logger.info(f"세그먼테이션 시작: 이미지 크기 {width}x{height}")
        
        # 1단계: 초기 세그먼트 생성 (Felzenszwalb 알고리즘 사용)
        # scale 매개변수를 Felzenszwalb의 scale로 변환
        adapted_scale = int(self.scale * 0.8)
        min_size = int(self.scale * 0.5)
        
        logger.info(f"초기 세그먼테이션 시작: adapted_scale={adapted_scale}, min_size={min_size}")
        segments = felzenszwalb(
            image, 
            scale=adapted_scale,
            sigma=0.8,
            min_size=min_size
        )
        
        # 세그먼트 개수 확인
        num_segments = len(np.unique(segments))
        logger.info(f"초기 세그먼트 생성 완료: 세그먼트 수 {num_segments}")
        
        # 2단계: 세그먼트 경계 개선 (형태 및 컴팩트니스 고려)
        # RAG(Region Adjacency Graph) 생성
        rag = graph.rag_mean_color(image, segments)
        
        # 병합 기준 설정: 형태와 컴팩트니스 가중치 고려
        merged_segments = graph.cut_threshold(
            segments, 
            rag, 
            self._merge_criterion_hybrid,
            in_place=False
        )
        
        # 최종 세그먼트 개수 확인
        num_final_segments = len(np.unique(merged_segments))
        logger.info(f"세그먼트 병합 완료: 최종 세그먼트 수 {num_final_segments}")
        
        # 세그먼트 맵 저장
        self.segment_map = merged_segments
        
        # 3단계: 세그먼트 속성 계산 및 객체 생성
        self._extract_segment_properties()
        
        return self.segment_map, self.segments
    
    def _merge_criterion_hybrid(self, graph, src, dst):
        """
        세그먼트 병합 기준 함수
        - 색상, 형태, 컴팩트니스를 고려한 하이브리드 기준
        
        Args:
            graph: RAG 그래프
            src, dst: 병합 대상 노드 ID
            
        Returns:
            float: 병합 점수 (낮을수록 병합 가능성 높음)
        """
        # 색상 차이 계산 (스펙트럼 기준)
        color_diff = abs(graph.nodes[src]['mean color'] - graph.nodes[dst]['mean color'])
        
        # 형태 차이 계산 (면적과 둘레 비율 기반)
        src_area = graph.nodes[src]['pixel count']
        dst_area = graph.nodes[dst]['pixel count']
        
        # 둘레는 근사치로 계산 (실제 계산은 복잡함)
        src_perimeter = np.sqrt(4 * np.pi * src_area)
        dst_perimeter = np.sqrt(4 * np.pi * dst_area)
        
        # 형태 지수 계산 (둘레/면적 비율)
        src_shape_index = src_perimeter / (4 * np.sqrt(src_area))
        dst_shape_index = dst_perimeter / (4 * np.sqrt(dst_area))
        shape_diff = abs(src_shape_index - dst_shape_index)
        
        # 컴팩트니스 계산 (원형에 가까운 정도)
        src_compactness = (4 * np.pi * src_area) / (src_perimeter ** 2) if src_perimeter > 0 else 0
        dst_compactness = (4 * np.pi * dst_area) / (dst_perimeter ** 2) if dst_perimeter > 0 else 0
        compactness_diff = abs(src_compactness - dst_compactness)
        
        # 종합 점수 계산 (낮을수록 병합 가능성 높음)
        color_weight = 1 - self.shape_weight
        shape_component = self.shape_weight * (
            (1 - self.compactness_weight) * shape_diff + 
            self.compactness_weight * compactness_diff
        )
        
        score = color_weight * color_diff + shape_component
        
        # 스케일 매개변수를 기준으로 병합 결정
        # 스케일이 클수록 더 많은 병합 허용 (점수 기준값 증가)
        threshold = 0.1 + (self.scale / 1000)
        
        return score < threshold
    
    def _extract_segment_properties(self):
        """
        세그먼트 속성을 추출하고 Segment 객체 생성
        """
        logger.info("세그먼트 속성 추출 시작")
        
        # 세그먼트 레이블 목록
        unique_labels = np.unique(self.segment_map)
        
        # 세그먼트 속성 추출
        region_props = measure.regionprops(self.segment_map + 1)  # +1은 배경 0을 피하기 위함
        
        self.segments = []
        
        for i, props in enumerate(region_props):
            label = unique_labels[i]
            
            # 해당 세그먼트의 모든 픽셀 좌표 찾기
            pixels = [(coord[0], coord[1]) for coord in props.coords]
            
            # 세그먼트 내 픽셀들의 평균 색상 계산
            mask = (self.segment_map == label)
            mean_color = np.mean(self.input_image[mask]) if np.any(mask) else 0
            
            # 면적 및 둘레 계산
            area = props.area
            perimeter = props.perimeter
            
            # 형태 지수 계산
            shape_index = perimeter / (4 * np.sqrt(area)) if area > 0 else 0
            
            # 컴팩트니스 계산
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # 세그먼트 객체 생성
            segment = Segment(
                id=label,
                pixels=pixels,
                mean_color=mean_color,
                area=area,
                perimeter=perimeter,
                shape_index=shape_index,
                compactness=compactness
            )
            
            self.segments.append(segment)
        
        # 인접 세그먼트 계산
        self._compute_segment_neighbors()
        
        logger.info(f"세그먼트 속성 추출 완료: {len(self.segments)}개 세그먼트")
    
    def _compute_segment_neighbors(self):
        """
        각 세그먼트의 인접 세그먼트 계산
        """
        # RAG 생성으로 인접 관계 파악
        rag = graph.rag_mean_color(self.input_image, self.segment_map)
        
        # 각 세그먼트마다 인접 목록 저장
        for segment in self.segments:
            # RAG에서 해당 노드의 인접 노드 가져오기
            if segment.id in rag.adj:
                segment.neighbors = list(rag.adj[segment.id].keys())
    
    def visualize_segments(self, output_path=None):
        """
        세그먼테이션 결과 시각화
        
        Args:
            output_path (str, optional): 결과 저장 경로
        """
        if self.segment_map is None:
            logger.error("세그먼테이션이 수행되지 않았습니다. segment() 메서드를 먼저 호출하세요.")
            return
        
        # 랜덤 색상 생성
        unique_labels = np.unique(self.segment_map)
        num_segments = len(unique_labels)
        colors = np.random.randint(0, 255, size=(num_segments, 3), dtype=np.uint8)
        
        # 레이블에 따라 이미지에 색상 할당
        result_img = np.zeros((self.segment_map.shape[0], self.segment_map.shape[1], 3), dtype=np.uint8)
        
        for i, label in enumerate(unique_labels):
            mask = (self.segment_map == label)
            result_img[mask] = colors[i]
        
        # 원본 이미지와 결과 시각화
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 3, 1)
        plt.title("원본 이미지")
        plt.imshow(self.input_image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("세그먼트 맵")
        plt.imshow(self.segment_map, cmap='nipy_spectral')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(f"세그먼테이션 결과 (세그먼트 수: {num_segments})")
        plt.imshow(result_img)
        plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"세그먼테이션 결과 시각화 저장 완료: {output_path}")
        
        plt.show()
    
    def get_boundary_segments(self, threshold=0.8):
        """
        경계선 특성을 가진 세그먼트만 반환
        
        Args:
            threshold (float): 경계선으로 판단할 형태 지수 임계값 (기본값: 0.8)
            
        Returns:
            List[Segment]: 경계선으로 판단되는 세그먼트 목록
        """
        if not self.segments:
            logger.error("세그먼테이션이 수행되지 않았습니다. segment() 메서드를 먼저 호출하세요.")
            return []
        
        boundary_segments = []
        
        for segment in self.segments:
            # 형태 지수가 높고 (가늘고 긴 형태)
            # 컴팩트니스가 낮은 (원형과 거리가 먼) 세그먼트를 경계선으로 판단
            if segment.shape_index > threshold and segment.compactness < (1 - threshold / 2):
                boundary_segments.append(segment)
        
        logger.info(f"경계선 세그먼트 추출 완료: 전체 {len(self.segments)}개 중 {len(boundary_segments)}개 식별")
        return boundary_segments
    
    def extract_boundary_mask(self, threshold=0.8):
        """
        경계선 세그먼트를 마스크로 추출
        
        Args:
            threshold (float): 경계선으로 판단할 형태 지수 임계값
            
        Returns:
            numpy.ndarray: 경계선 마스크 (이진 이미지)
        """
        if self.segment_map is None:
            logger.error("세그먼테이션이 수행되지 않았습니다. segment() 메서드를 먼저 호출하세요.")
            return None
        
        # 경계선 세그먼트 가져오기
        boundary_segments = self.get_boundary_segments(threshold)
        
        # 빈 마스크 생성
        boundary_mask = np.zeros_like(self.segment_map, dtype=np.uint8)
        
        # 경계선 세그먼트의 픽셀을 마스크에 표시
        for segment in boundary_segments:
            for pixel in segment.pixels:
                boundary_mask[pixel] = 255
        
        return boundary_mask
    
def segment(image, scale=50, shape=0.3, compactness=0.5):
    """
    다중해상도 세그먼테이션 수행 (편의 함수)
    
    Args:
        image (numpy.ndarray): 입력 이미지
        scale (int): 세그먼트 크기를 제어하는 스케일 매개변수 (기본값: 50)
        shape (float): 형태의 중요도 (0~1 사이, 기본값: 0.3)
        compactness (float): 컴팩트니스의 중요도 (0~1 사이, 기본값: 0.5)
        
    Returns:
        Tuple[numpy.ndarray, List[Segment]]: 세그먼트 맵과 세그먼트 객체 목록
    """
    segmenter = MultiResolutionSegmentation(scale, shape, compactness)
    return segmenter.segment(image)

if __name__ == "__main__":
    # 테스트용 코드
    from src.preprocessing import image_preprocessing
    
    # 테스트 이미지 경로
    test_image_path = "data/sample/raw_cadastral_maps/test_map.tif"
    output_path = "data/results/segmented/test_map_segmented.png"
    
    # 이미지 전처리 및 세그먼테이션 실행
    try:
        # 이미지 로드 및 전처리
        processed_image = image_preprocessing.preprocess(test_image_path)
        
        # 다중해상도 세그먼테이션 수행
        segmenter = MultiResolutionSegmentation(scale=75, shape_weight=0.3, compactness_weight=0.5)
        segment_map, segments = segmenter.segment(processed_image)
        
        # 결과 시각화
        segmenter.visualize_segments(output_path)
        
        # 경계선 추출
        boundary_mask = segmenter.extract_boundary_mask(threshold=0.7)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("원본 이미지")
        plt.imshow(processed_image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("추출된 경계선")
        plt.imshow(boundary_mask, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("data/results/segmented/test_map_boundaries.png", dpi=300)
        plt.show()
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}")
