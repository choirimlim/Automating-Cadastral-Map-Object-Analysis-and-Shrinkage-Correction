"""
아핀 변환 기반 신축량 보정 모듈
- 지적원도의 신축량(왜곡)을 보정하기 위한 변환 기능 제공
- 지상기준점(GCP)을 이용한 아핀 변환 및 투영 변환 구현
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Union, Optional
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import os
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AffineCorrection:
    """
    아핀 변환을 이용한 지적원도 신축량 보정 클래스
    - 지상기준점(GCP)을 이용해 원도의 왜곡을 보정
    """
    
    def __init__(self, method='affine'):
        """
        초기화
        
        Args:
            method (str): 변환 방법 ('affine' 또는 'projective'), 기본값 'affine'
        """
        self.method = method
        self.transformation_matrix = None
        self.src_points = None
        self.dst_points = None
        self.rmse = None
        logger.info(f"신축량 보정 초기화: 변환 방법 {method}")
    
    def set_control_points(self, src_points, dst_points):
        """
        지상기준점(GCP) 설정
        
        Args:
            src_points (List[Tuple]): 원본 이미지의 좌표점 [(x1, y1), (x2, y2), ...]
            dst_points (List[Tuple]): 참조 좌표계의 좌표점 [(x1, y1), (x2, y2), ...]
            
        Returns:
            bool: 설정 성공 여부
        """
        if len(src_points) < 3:
            logger.error("아핀 변환에는 최소 3개의 기준점이 필요합니다.")
            return False
            
        if len(src_points) != len(dst_points):
            logger.error(f"원본 좌표({len(src_points)}개)와 참조 좌표({len(dst_points)}개)의 개수가 일치하지 않습니다.")
            return False
        
        self.src_points = np.array(src_points, dtype=np.float32)
        self.dst_points = np.array(dst_points, dtype=np.float32)
        logger.info(f"기준점 설정 완료: {len(src_points)}개 GCP")
        
        return True
    
    def calculate_transform(self):
        """
        변환 행렬 계산
        
        Returns:
            numpy.ndarray: 변환 행렬
        """
        if self.src_points is None or self.dst_points is None:
            logger.error("기준점이 설정되지 않았습니다. set_control_points() 메서드를 먼저 호출하세요.")
            return None
        
        if self.method == 'affine':
            # 아핀 변환 행렬 계산
            self.transformation_matrix, inliers = cv2.estimateAffinePartial2D(
                self.src_points, self.dst_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            logger.info(f"아핀 변환 행렬 계산 완료 (내부 점 수: {np.sum(inliers)})")
        
        elif self.method == 'projective':
            # 투영 변환 행렬 계산 (원근 변환)
            if len(self.src_points) < 4:
                logger.error("투영 변환에는 최소 4개의 기준점이 필요합니다.")
                return None
                
            self.transformation_matrix, inliers = cv2.findHomography(
                self.src_points, self.dst_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            logger.info(f"투영 변환 행렬 계산 완료 (내부 점 수: {np.sum(inliers)})")
        
        else:
            logger.error(f"지원되지 않는 변환 방법입니다: {self.method}")
            return None
        
        # RMSE 계산
        transformed_points = self.transform_points(self.src_points)
        self.rmse = np.sqrt(np.mean(np.sum((transformed_points - self.dst_points) ** 2, axis=1)))
        logger.info(f"변환 RMSE: {self.rmse:.4f}")
        
        return self.transformation_matrix
    
    def transform_points(self, points):
        """
        좌표점들을 변환
        
        Args:
            points (numpy.ndarray): 변환할 좌표점들
            
        Returns:
            numpy.ndarray: 변환된 좌표점들
        """
        if self.transformation_matrix is None:
            logger.error("변환 행렬이 계산되지 않았습니다. calculate_transform() 메서드를 먼저 호출하세요.")
            return points
        
        points = np.array(points, dtype=np.float32)
        
        if self.method == 'affine':
            # 아핀 변환 적용
            transformed = cv2.transform(points.reshape(-1, 1, 2), self.transformation_matrix)
            return transformed.reshape(-1, 2)
        
        elif self.method == 'projective':
            # 투영 변환 적용
            points_homogeneous = np.c_[points, np.ones(len(points))]
            transformed_homogeneous = np.dot(self.transformation_matrix, points_homogeneous.T).T
            transformed = transformed_homogeneous[:, :2] / transformed_homogeneous[:, 2:3]
            return transformed
        
        else:
            logger.error(f"지원되지 않는 변환 방법입니다: {self.method}")
            return points
    
    def apply_correction(self, image, output_size=None):
        """
        이미지에 신축량 보정 적용
        
        Args:
            image (numpy.ndarray): 입력 이미지
            output_size (tuple, optional): 출력 이미지 크기(너비, 높이), 기본값 None(원본 크기 유지)
            
        Returns:
            numpy.ndarray: 보정된 이미지
        """
        if self.transformation_matrix is None:
            logger.error("변환 행렬이 계산되지 않았습니다. calculate_transform() 메서드를 먼저 호출하세요.")
            return image
        
        height, width = image.shape[:2]
        if output_size is None:
            output_size = (width, height)
        
        if self.method == 'affine':
            # 아핀 변환 적용
            corrected = cv2.warpAffine(
                image, self.transformation_matrix, output_size,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        
        elif self.method == 'projective':
            # 투영 변환 적용
            corrected = cv2.warpPerspective(
                image, self.transformation_matrix, output_size,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        
        else:
            logger.error(f"지원되지 않는 변환 방법입니다: {self.method}")
            return image
        
        logger.info(f"이미지 신축량 보정 완료: 출력 크기 {output_size}")
        return corrected
    
    def apply_correction_to_vector(self, geometry_data, crs=None):
        """
        벡터 데이터에 신축량 보정 적용
        
        Args:
            geometry_data (GeoDataFrame/Shapely Geometry): 보정할 벡터 데이터
            crs (dict/str, optional): 좌표 참조 시스템, 기본값 None
            
        Returns:
            GeoDataFrame/Shapely Geometry: 보정된 벡터 데이터
        """
        if self.transformation_matrix is None:
            logger.error("변환 행렬이 계산되지 않았습니다. calculate_transform() 메서드를 먼저 호출하세요.")
            return geometry_data
        
        # GeoDataFrame 처리
        if isinstance(geometry_data, gpd.GeoDataFrame):
            # 새 GeoDataFrame 생성
            corrected_gdf = geometry_data.copy()
            
            # 각 지오메트리 보정
            corrected_geometries = []
            for geom in geometry_data.geometry:
                corrected_geom = self._transform_geometry(geom)
                corrected_geometries.append(corrected_geom)
            
            # 보정된 지오메트리로 업데이트
            corrected_gdf.geometry = corrected_geometries
            
            # CRS 설정
            if crs is not None:
                corrected_gdf.crs = crs
            
            logger.info(f"벡터 데이터 보정 완료: {len(corrected_gdf)}개 객체")
            return corrected_gdf
        
        # 단일 Shapely 지오메트리 처리
        else:
            logger.info(f"단일 지오메트리 보정 완료: {type(geometry_data).__name__}")
            return self._transform_geometry(geometry_data)
    
    def _transform_geometry(self, geometry):
        """
        Shapely 지오메트리 객체 변환
        
        Args:
            geometry: Shapely 지오메트리 객체
            
        Returns:
            변환된 지오메트리 객체
        """
        # Point 처리
        if isinstance(geometry, Point):
            coords = np.array([[geometry.x, geometry.y]])
            transformed = self.transform_points(coords)[0]
            return Point(transformed)
        
        # LineString 처리
        elif isinstance(geometry, LineString):
            coords = np.array(geometry.coords)
            transformed = self.transform_points(coords)
            return LineString(transformed)
        
        # Polygon 처리
        elif isinstance(geometry, Polygon):
            exterior_coords = np.array(geometry.exterior.coords)
            transformed_exterior = self.transform_points(exterior_coords)
            
            # 내부 링 처리
            transformed_interiors = []
            for interior in geometry.interiors:
                interior_coords = np.array(interior.coords)
                transformed_interior = self.transform_points(interior_coords)
                transformed_interiors.append(transformed_interior)
            
            return Polygon(transformed_exterior, transformed_interiors)
        
        # 다른 유형의 지오메트리는 원본 반환
        else:
            logger.warning(f"지원되지 않는 지오메트리 유형: {type(geometry)}")
            return geometry