# 지적원도 객체분석 및 신축량 보정 자동화 방법론

본 문서는 지적원도의 객체분석 및 신축량 보정을 자동화하기 위한 방법론을 설명합니다.

## 1. 개요

지적원도(cadastral map)는 토지 등록 및 관리를 위한 기본 자료로, 토지의 위치, 형태, 경계 등을 나타내는 지도입니다. 종이 지적원도의 디지털화와 자동 분석은 다음과 같은 문제점을 해결하고자 합니다:

1. **지적원도의 변형**: 종이 지적원도는 시간이 지남에 따라 신축(伸縮, 늘어나거나 줄어드는 현상)이 발생하여 실제 지형과 불일치가 발생합니다.
2. **수작업의 비효율성**: 지적원도 디지털화 과정에서 경계선 추출 등의 작업이 수작업으로 이루어져 시간과 노력이 많이 소요됩니다.
3. **일관성 부족**: 수작업에 의한 디지털화는 작업자에 따라 결과물의 품질 차이가 발생할 수 있습니다.

이러한 문제점을 해결하기 위해 컴퓨터 비전과 기계학습 기술을 활용한 자동화 방법론을 개발하였습니다.

## 2. 방법론 개요

본 프로젝트의 처리 파이프라인은 다음과 같은 4단계로 구성됩니다:

1. **이미지 전처리**: 노이즈 제거 및 이미지 품질 향상
2. **객체 세그먼테이션**: 다중해상도 세그먼테이션을 통한 토지 구획 식별
3. **경계 추출**: 기계학습 기반 경계 검출
4. **신축량 보정**: 지상기준점(GCP) 기반 변환 적용

## 3. 세부 방법론

### 3.1 이미지 전처리

전처리 단계는 스캔된 지적원도 이미지의 품질을 향상시키고 후속 분석을 위한 준비 과정입니다.

#### 주요 기법:

1. **그레이스케일 변환**: 컬러 이미지를 그레이스케일로 변환하여 처리를 단순화합니다.
2. **가우시안 블러(Gaussian Blur)**: 고주파 노이즈를 제거하여 이미지를 부드럽게 만듭니다.
   ```python
   blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
   ```
3. **적응형 히스토그램 평활화(CLAHE)**: 이미지의 대비를 향상시켜 희미한 경계선도 잘 보이게 합니다.
   ```python
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
   enhanced = clahe.apply(blurred)
   ```
4. **적응형 이진화(Adaptive Thresholding)**: 지역적 임계값을 적용하여 이미지를 이진화합니다.
   ```python
   binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
   ```
5. **모폴로지 연산**: 작은 노이즈를 제거하고 끊어진 선을 연결합니다.
   ```python
   kernel = np.ones((3, 3), np.uint8)
   processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
   ```

### 3.2 객체 세그먼테이션

객체 세그먼테이션은 이미지를 의미 있는 영역으로 분할하는 과정입니다. 다중해상도 세그먼테이션(MRS)은 eCognition 소프트웨어에서 널리 사용되는 기법으로, 픽셀 단위가 아닌 객체 기반 이미지 분석(OBIA)을 가능하게 합니다.

#### MRS 알고리즘의 주요 특징:

1. **스케일 매개변수**: 세그먼트 크기를 제어합니다. 값이 클수록 더 큰 세그먼트가 생성됩니다.
2. **형태(Shape) 가중치**: 형태의 중요도를 결정합니다. 값이 클수록 형태적 특성이 더 중요해집니다.
3. **컴팩트니스(Compactness) 가중치**: 컴팩트니스의 중요도를 결정합니다. 값이 클수록 더 컴팩트한 형태를 선호합니다.

본 프로젝트에서는 MRS를 위해 Felzenszwalb 알고리즘을 기반으로 한 구현과 Region Adjacency Graph(RAG)를 이용한 세그먼트 병합을 적용했습니다:

```python
# 초기 세그먼테이션
segments = felzenszwalb(image, scale=adapted_scale, sigma=0.8, min_size=min_size)

# RAG 생성 및 세그먼트 병합
rag = graph.rag_mean_color(image, segments)
merged_segments = graph.cut_threshold(segments, rag, merge_criterion)
```

### 3.3 경계 추출

세그먼트화된 이미지에서 경계선을 추출하기 위해 Support Vector Machine(SVM) 분류기를 활용했습니다. 이 분류기는 각 세그먼트가 경계선인지 아닌지를 판별합니다.

#### 경계 추출을 위한 특성:

1. **스펙트럼 특성**:
   - 평균 강도(mean intensity)
   - 표준 편차(standard deviation)
   - 이웃과의 대비(contrast with neighbors)

2. **기하학적 특성**:
   - 형태 지수(shape index): 둘레와 면적의 관계를 나타내는 지표
   - 컴팩트니스(compactness): 객체가 원형에 얼마나 가까운지를 나타내는 지표
   - 장축비(elongation): 객체의 길이와 폭의 비율

#### SVM 분류기:

```python
classifier = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
classifier.fit(X_train_scaled, y_train)
```

### 3.4 신축량 보정

지적원도의 신축량을 보정하기 위해 아핀 변환(Affine Transformation) 또는 투영 변환(Projective Transformation)을 적용합니다. 이 과정에서 지상기준점(GCP)을 활용합니다.

#### 보정 과정:

1. **GCP 설정**: 원본 좌표와 참조 좌표 간의 대응점을 설정합니다.
2. **변환 행렬 계산**: 설정된 GCP를 기반으로, 가장 적합한 변환 행렬을 계산합니다.
   ```python
   transformation_matrix, inliers = cv2.estimateAffinePartial2D(
       src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
   )
   ```
3. **변환 적용**: 계산된 변환 행렬을 이용해 이미지 변환을 수행합니다.
   ```python
   corrected = cv2.warpAffine(
       image, transformation_matrix, output_size,
       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
   )
   ```

## 4. 성능 평가

### 4.1 객체 분할 및 경계 추출 평가

- **정확도(Accuracy)**: 전체 세그먼트 중 올바르게 분류된 세그먼트의 비율
- **정밀도(Precision)**: 경계로 분류된 세그먼트 중 실제 경계인 세그먼트의 비율
- **재현율(Recall)**: 실제 경계 세그먼트 중 경계로 분류된 세그먼트의 비율
- **F1 점수**: 정밀도와 재현율의 조화 평균

### 4.2 신축량 보정 평가

- **RMSE(Root Mean Square Error)**: 보정 후 GCP의 예측 위치와 실제 위치 간의 오차
- **최대 오차**: 가장 큰 위치 오차
- **시각적 평가**: 보정 전후 이미지 비교를 통한 품질 평가

## 5. 결론 및 향후 연구

본 방법론은 지적원도의 객체분석 및 신축량 보정을 자동화하여 작업 효율성을 크게 향상시킵니다. 특히 컴퓨터 비전과 기계학습 기술을 활용함으로써 일관된 품질의 결과물을 얻을 수 있습니다.

향후 연구 방향으로는 다음과 같은 내용을 고려하고 있습니다:

1. **딥러닝 모델 적용**: U-Net과 같은 딥러닝 모델을 활용하여 경계 추출 정확도 향상
2. **다양한 원도 유형 지원**: 다양한 형태와 상태의 지적원도에 대한 처리 능력 향상
3. **자동화 수준 제고**: 사용자 개입을 최소화하는 엔드-투-엔드 자동화 시스템 개발

## 참고 문헌

1. Blaschke, T. (2010). Object based image analysis for remote sensing. ISPRS Journal of Photogrammetry and Remote Sensing, 65(1), 2-16.
2. Felzenszwalb, P. F., & Huttenlocher, D. P. (2004). Efficient graph-based image segmentation. International Journal of Computer Vision, 59(2), 167-181