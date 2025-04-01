"""
SVM 기반 경계 추출 모듈
- 지적원도에서 토지 구획의 경계를 추출하기 위한 분류기
- 스펙트럼 및 기하학적 특성을 활용하여 경계/비경계 분류 수행
"""

import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
import joblib

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BoundaryFeatures:
    """경계 추출을 위한 특성 클래스"""
    segment_id: int
    mean_intensity: float
    std_intensity: float
    shape_index: float
    compactness: float
    elongation: float
    neighbor_contrast: float
    is_boundary: Optional[bool] = None

class SVMBoundaryClassifier:
    """
    SVM을 사용한 경계/비경계 분류기
    - 지적원도의 세그먼트에서 경계선을 식별하는 모델
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        초기화
        
        Args:
            kernel (str): SVM 커널 유형 (linear, poly, rbf, sigmoid)
            C (float): 규제 파라미터
            gamma (str/float): 커널 계수
        """
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'mean_intensity', 'std_intensity', 'shape_index', 
            'compactness', 'elongation', 'neighbor_contrast'
        ]
        logger.info(f"SVM 경계 분류기 초기화: kernel={kernel}, C={C}, gamma={gamma}")
    
    def _extract_features(self, segments, image=None):
        """
        세그먼트에서 특성을 추출
        
        Args:
            segments (List): 세그먼트 객체 목록
            image (numpy.ndarray, optional): 원본 이미지
            
        Returns:
            List[BoundaryFeatures]: 추출된 특성 목록
        """
        features_list = []
        
        for segment in segments:
            # 기본 특성 추출
            mean_intensity = segment.mean_color
            
            # 세그먼트 내 픽셀 강도의 표준 편차 계산
            if image is not None:
                pixel_intensities = [image[px[0], px[1]] for px in segment.pixels]
                std_intensity = np.std(pixel_intensities)
            else:
                std_intensity = 0
            
            # 형태 관련 특성
            shape_index = segment.shape_index
            compactness = segment.compactness
            
            # 장축비(elongation) 계산 (근사값)
            # 실제로는 PCA나 최소 영역 사각형을 사용하여 계산 가능
            elongation = shape_index * (1 - compactness) if compactness < 1 else 0
            
            # 이웃과의 대비(contrast) 계산
            neighbor_contrast = 0
            if segment.neighbors and len(segment.neighbors) > 0:
                neighbors_mean = sum(seg.mean_color for seg in segments if seg.id in segment.neighbors)
                if len(segment.neighbors) > 0:
                    neighbors_mean /= len(segment.neighbors)
                    neighbor_contrast = abs(mean_intensity - neighbors_mean)
            
            # 특성 객체 생성
            features = BoundaryFeatures(
                segment_id=segment.id,
                mean_intensity=mean_intensity,
                std_intensity=std_intensity,
                shape_index=shape_index,
                compactness=compactness,
                elongation=elongation,
                neighbor_contrast=neighbor_contrast
            )
            
            features_list.append(features)
        
        return features_list
    
    def _features_to_array(self, features_list):
        """
        특성 객체 목록을 NumPy 배열로 변환
        
        Args:
            features_list (List[BoundaryFeatures]): 특성 객체 목록
        
        Returns:
            numpy.ndarray: 특성 배열
        """
        X = np.array([
            [
                f.mean_intensity, f.std_intensity, f.shape_index,
                f.compactness, f.elongation, f.neighbor_contrast
            ]
            for f in features_list
        ])
        return X
    
    def train(self, segments, labels, image=None):
        """
        SVM 모델 학습
        
        Args:
            segments (List): 세그먼트 객체 목록
            labels (List[bool]): 각 세그먼트가 경계인지 여부 (True/False)
            image (numpy.ndarray, optional): 원본 이미지
            
        Returns:
            Dict: 학습 결과 정보
        """
        logger.info(f"총 {len(segments)}개 세그먼트로 모델 학습 시작")
        
        # 특성 추출
        features_list = self._extract_features(segments, image)
        
        # 각 특성에 레이블 할당
        for i, features in enumerate(features_list):
            if i < len(labels):
                features.is_boundary = labels[i]
        
        # 특성 배열과 레이블 배열 생성
        X = self._features_to_array(features_list)
        y = np.array(labels)
        
        # 학습 및 테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 모델 학습
        self.model.fit(X_train_scaled, y_train)
        
        # 평가
        y_pred = self.model.predict(X_test_scaled)
        
        # 성능 지표 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"모델 학습 완료: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        self.is_trained = True
        
        # 특성 중요도 계산 (커널이 linear일 경우에만)
        feature_importance = {}
        if self.model.kernel == 'linear':
            for i, importance in enumerate(self.model.coef_[0]):
                feature_importance[self.feature_names[i]] = importance
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'feature_importance': feature_importance
        }
    
    def predict(self, segments, image=None):
        """
        세그먼트에 대한 경계/비경계 예측
        
        Args:
            segments (List): 세그먼트 객체 목록
            image (numpy.ndarray, optional): 원본 이미지
            
        Returns:
            List[bool]: 각 세그먼트에 대한 경계 여부 (True/False)
        """
        if not self.is_trained:
            logger.error("모델이 학습되지 않았습니다. train() 메서드를 먼저 호출하세요.")
            return []
        
        # 특성 추출
        features_list = self._extract_features(segments, image)
        
        # 특성 배열 생성
        X = self._features_to_array(features_list)
        
        # 특성 스케일링
        X_scaled = self.scaler.transform(X)
        
        # 예측
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]  # 경계일 확률
        
        logger.info(f"예측 완료: {len(segments)}개 세그먼트 중 {sum(predictions)}개가 경계로 분류됨")
        
        # 각 세그먼트에 예측 결과와 확률 할당
        for i, feature in enumerate(features_list):
            feature.is_boundary = bool(predictions[i])
        
        return predictions, probabilities, features_list
    
    def save_model(self, model_path):
        """
        모델과 스케일러 저장
        
        Args:
            model_path (str): 모델 저장 경로
        """
        if not self.is_trained:
            logger.error("학습되지 않은 모델은 저장할 수 없습니다.")
            return False
        
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델과 스케일러 저장
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }, model_path)
        
        logger.info(f"모델 저장 완료: {model_path}")
        return True
    
    def load_model(self, model_path):
        """
        저장된 모델과 스케일러 로드
        
        Args:
            model_path (str): 모델 파일 경로
        """
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']
            
            logger.info(f"모델 로드 완료: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False
    
    def visualize_predictions(self, image, segments, predictions, output_path=None):
        """
        경계 예측 결과 시각화
        
        Args:
            image (numpy.ndarray): 원본 이미지
            segments (List): 세그먼트 객체 목록
            predictions (List[bool]): 각 세그먼트의 경계 여부 예측
            output_path (str, optional): 시각화 결과 저장 경로
        """
        if len(segments) != len(predictions):
            logger.error(f"세그먼트 수({len(segments)})와 예측 수({len(predictions)})가 일치하지 않습니다.")
            return
        
        # 결과 이미지 생성
        result_img = np.zeros_like(image) if len(image.shape) == 3 else np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        
        # 원본 이미지가 그레이스케일이면 3채널로 변환
        if len(image.shape) == 2:
            orig_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            orig_img = image.copy()
        
        # 각 세그먼트에 색상 적용 (경계: 빨간색, 비경계: 원본)
        for i, segment in enumerate(segments):
            if i < len(predictions):
                for pixel in segment.pixels:
                    if 0 <= pixel[0] < result_img.shape[0] and 0 <= pixel[1] < result_img.shape[1]:
                        if predictions[i]:  # 경계로 예측된 경우
                            result_img[pixel[0], pixel[1]] = [0, 0, 255]  # 빨간색
                        else:  # 비경계로 예측된 경우
                            result_img[pixel[0], pixel[1]] = orig_img[pixel[0], pixel[1]]
        
        # 결과 시각화
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 3, 1)
        plt.title("원본 이미지")
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title(f"경계 예측 결과 (경계 세그먼트: {sum(predictions)}개)")
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # 경계만 표시한 이미지
        boundary_only = np.zeros_like(result_img)
        for i, segment in enumerate(segments):
            if i < len(predictions) and predictions[i]:
                for pixel in segment.pixels:
                    if 0 <= pixel[0] < boundary_only.shape[0] and 0 <= pixel[1] < boundary_only.shape[1]:
                        boundary_only[pixel[0], pixel[1]] = [255, 255, 255]
        
        plt.subplot(1, 3, 3)
        plt.title("추출된 경계선")
        plt.imshow(cv2.cvtColor(boundary_only, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"예측 결과 시각화 저장 완료: {output_path}")
        
        plt.show()
        
        # 추출된 경계선 반환
        return cv2.cvtColor(boundary_only, cv2.COLOR_BGR2GRAY)


def extract_boundaries(segmented_data, model_path=None, train_data=None):
    """
    SVM 분류기를 사용하여 세그먼트에서 경계를 추출하는 편의 함수
    
    Args:
        segmented_data (Tuple): (세그먼트 맵, 세그먼트 객체 목록)
        model_path (str, optional): 학습된 모델 경로 (없으면 기본 모델 사용)
        train_data (Tuple, optional): 학습 데이터 (세그먼트 목록, 레이블 목록)
        
    Returns:
        numpy.ndarray: 추출된 경계선 이미지 (이진 마스크)
    """
    segment_map, segments = segmented_data
    classifier = SVMBoundaryClassifier()
    
    # 기존 모델 로드 또는 새로 학습
    if model_path and Path(model_path).exists():
        logger.info(f"모델 로드: {model_path}")
        classifier.load_model(model_path)
    elif train_data:
        logger.info("제공된 데이터로 모델 학습")
        train_segments, train_labels = train_data
        classifier.train(train_segments, train_labels)
        if model_path:
            classifier.save_model(model_path)
    else:
        logger.warning("모델 경로가 없고 학습 데이터도 제공되지 않았습니다. 기본 파라미터로 작업합니다.")
        # 기본 학습 데이터를 생성하거나 사전 정의된 규칙 기반 분류 사용
        # 여기서는 형태 지수와 컴팩트니스를 기반으로 한 간단한 규칙 사용
        predictions = np.array([segment.shape_index > 0.8 and segment.compactness < 0.4 for segment in segments])
        return classifier.visualize_predictions(np.zeros_like(segment_map), segments, predictions)
    
    # 경계 예측
    predictions, _, _ = classifier.predict(segments)
    
    # 예측 결과 시각화 및 경계선 추출
    boundary_image = classifier.visualize_predictions(np.zeros_like(segment_map), segments, predictions)
    
    return boundary_image


if __name__ == "__main__":
    # 테스트용 코드
    from src.preprocessing import image_preprocessing
    from src.segmentation import multiresolution
    import numpy as np
    
    # 테스트 이미지 경로
    test_image_path = "data/sample/raw_cadastral_maps/test_map.tif"
    model_save_path = "data/models/boundary_classifier.pkl"
    output_path = "data/results/boundaries/test_map_boundaries.png"
    
    try:
        # 이미지 로드 및 전처리
        processed_image = image_preprocessing.preprocess(test_image_path)
        
        # 다중해상도 세그먼테이션 수행
        segmenter = multiresolution.MultiResolutionSegmentation(scale=75, shape_weight=0.3, compactness_weight=0.5)
        segment_map, segments = segmenter.segment(processed_image)
        
        # 학습 데이터 생성 (실제로는 수동 레이블링 또는 참조 데이터 사용)
        # 여기서는 예시를 위해 형태 지수와 컴팩트니스 기반으로 가상 레이블 생성
        labels = []
        for segment in segments:
            # 가늘고 긴 형태(높은 형태 지수)와 낮은 컴팩트니스를 가진 세그먼트를 경계로 간주
            is_boundary = segment.shape_index > 0.8 and segment.compactness < 0.4
            labels.append(is_boundary)
        
        # SVM 분류기 학습
        classifier = SVMBoundaryClassifier(kernel='rbf', C=1.0, gamma='scale')
        train_result = classifier.train(segments, labels, processed_image)
        
        # 모델 저장
        classifier.save_model(model_save_path)
        
        # 예측 및 시각화
        predictions, probabilities, features = classifier.predict(segments, processed_image)
        boundary_image = classifier.visualize_predictions(processed_image, segments, predictions, output_path)
        
        # 편의 함수 테스트
        simple_boundary = extract_boundaries((segment_map, segments), model_save_path)
        
        # 결과 비교
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("SVM 분류기 결과")
        plt.imshow(boundary_image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("편의 함수 결과")
        plt.imshow(simple_boundary, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("data/results/boundaries/comparison.png", dpi=300)
        plt.show()
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}")