# 지적원도 객체분석 및 신축량 보정 자동화 시스템 사용 가이드

## 목차

1. [시작하기](#1-시작하기)
2. [명령줄 인터페이스](#2-명령줄-인터페이스)
3. [파이썬 API](#3-파이썬-api)
4. [주요 기능 설명](#4-주요-기능-설명)
5. [사용 사례](#5-사용-사례)
6. [GCP 파일 형식](#6-gcp-파일-형식)
7. [문제 해결](#7-문제-해결)

## 1. 시작하기

### 시스템 요구사항

- Python 3.8 이상
- GDAL 3.4.0 이상
- OpenCV 4.5.3 이상
- 기타 필수 패키지 (requirements.txt 참조)

### 설치 방법

1. 저장소 복제
```bash
git clone https://github.com/yourusername/cadastral-map-automation.git
cd cadastral-map-automation
```

2. 가상 환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

4. GDAL 설치 확인
```bash
python -c "from osgeo import gdal; print(gdal.__version__)"
```

## 2. 명령줄 인터페이스

### 기본 사용법

```bash
python src/main.py --input <입력_이미지_경로> --output <결과_저장_경로>
```

### 매개변수 설명

| 매개변수 | 설명 |
|----------|------|
| `--input`, `-i` | 입력 지적원도 이미지 경로 (필수) |
| `--output`, `-o` | 결과 저장 디렉토리 (기본값: data/results) |
| `--gcps`, `-g` | GCP(지상기준점) 파일 경로 |
| `--model`, `-m` | 경계 분류 모델 경로 |
| `--no-save` | 중간 결과 저장 안 함 |
| `--no-vis` | 결과 시각화 안 함 |

### 사용 예시

1. 기본 처리 (모든 중간 결과 및 시각화 포함)
```bash
python src/main.py --input data/sample/raw_cadastral_maps/map001.tif
```

2. GCP 파일 및 사전 학습된 모델 적용
```bash
python src/main.py --input data/sample/raw_cadastral_maps/map001.tif --gcps data/sample/ground_control_points/map001_gcps.txt --model data/models/boundary_classifier.pkl
```

3. 중간 결과 저장 및 시각화 없이 처리
```bash
python src/main.py --input data/sample/raw_cadastral_maps/map001.tif --no-save --no-vis
```

## 3. 파이썬 API

### 전체 파이프라인 실행

```python
from src.main import process_cadastral_map

# 전체 파이프라인 실행
result = process_cadastral_map(
    input_path="data/sample/raw_cadastral_maps/map001.tif",
    output_dir="data/results",
    gcps_file="data/sample/ground_control_points/map001_gcps.txt",
    model_path="data/models/boundary_classifier.pkl",
    save_intermediates=True,
    visualize=True
)

# 결과 확인
print(f"세그먼트 수: {result['segments_count']}")
```

### 개별 단계 실행

각 단계를 개별적으로 실행할 수도 있습니다:

```python
from src.preprocessing import image_preprocessing
from src.segmentation import multiresolution
from src.boundary_detection import svm_classifier
from src.correction import affine_transform

# 1. 이미지 전처리
processed_image = image_preprocessing.preprocess("data/sample/raw_cadastral_maps/map001.tif")

# 2. 객체 세그먼테이션
segmenter = multiresolution.MultiResolutionSegmentation(scale=75, shape_weight=0.3, compactness_weight=0.5)
segment_map, segments = segmenter.segment(processed_image)

# 3. 경계 추출
classifier = svm_classifier.SVMBoundaryClassifier()
classifier.load_model("data/models/boundary_classifier.pkl")
predictions, probabilities, features = classifier.predict(segments, processed_image)
boundary_mask = classifier.visualize_predictions(processed_image, segments, predictions)

# 4. 신축량 보정
gcps = [
    ((100, 150), (423000.5, 634200.2)),
    ((250, 300), (423150.8, 634350.6)),
    ((400, 200), (423300.2, 634250.1)),
    ((150, 400), (423050.6, 634450.3))
]
corrector = affine_transform.AffineCorrection(method='affine')
corrector.set_control_points([p[0] for p in gcps], [p[1] for p in gcps])
corrector.calculate_transform()
corrected_image = corrector.apply_correction(boundary_mask)
```

## 4. 주요 기능 설명

### 이미지 전처리

- **전처리 매개변수 설정**: 이미지 특성에 따라 전처리 매개변수를 조정할 수 있습니다.

```python
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
processed_image = image_preprocessing.preprocess(input_path, params)
```

### 객체 세그먼테이션

- **세그먼테이션 매개변수 조정**: 지적원도의 특성에 맞게 매개변수를 조정할 수 있습니다.

```python
# 더 큰 객체를 생성하려면 scale 값 증가
segmenter = multiresolution.MultiResolutionSegmentation(scale=100, shape_weight=0.3, compactness_weight=0.5)

# 더 작은 객체를 생성하려면 scale 값 감소
segmenter = multiresolution.MultiResolutionSegmentation(scale=50, shape_weight=0.3, compactness_weight=0.5)

# 형태에 더 중점을 두려면 shape_weight 증가
segmenter = multiresolution.MultiResolutionSegmentation(scale=75, shape_weight=0.5, compactness_weight=0.5)
```

### 경계 추출

- **SVM 모델 학습**: 레이블이 지정된 데이터로 분류 모델을 학습시킬 수 있습니다.

```python
# 학습 데이터 생성 (실제로는 수동 레이블링 또는 참조 데이터 사용)
labels = [True if segment.shape_index > 0.8 and segment.compactness < 0.4 else False for segment in segments]

# 모델 학습
classifier = svm_classifier.SVMBoundaryClassifier(kernel='rbf', C=1.0, gamma='scale')
train_result = classifier.train(segments, labels, processed_image)

# 모델 저장
classifier.save_model("data/models/my_boundary_classifier.pkl")
```

### 신축량 보정

- **변환 방법 선택**: 아핀 변환 또는 투영 변환을 선택할 수 있습니다.

```python
# 아핀 변환 (회전, 크기 조정, 이동)
corrector = affine_transform.AffineCorrection(method='affine')

# 투영 변환 (원근감 변환)
corrector = affine_transform.AffineCorrection(method='projective')
```

- **GCP 내보내기**: GCP 정보를 파일로 저장할 수 있습니다.

```python
corrector.export_gcps_to_file("data/results/corrected/gcps.txt")
```

## 5. 사용 사례

### 사례 1: 대량의 지적원도 일괄 처리

여러 지적원도를 일괄 처리하려면 다음과 같은 스크립트를 사용할 수 있습니다:

```python
import os
from src.main import process_cadastral_map

input_dir = "data/sample/raw_cadastral_maps"
output_dir = "data/results"
model_path = "data/models/boundary_classifier.pkl"

for file_name in os.listdir(input_dir):
    if file_name.endswith(('.tif', '.jpg', '.png')):
        input_path = os.path.join(input_dir, file_name)
        gcps_file = os.path.join("data/sample/ground_control_points", file_name.replace('.tif', '_gcps.txt'))
        
        if not os.path.exists(gcps_file):
            gcps_file = None
        
        print(f"처리 중: {file_name}")
        process_cadastral_map(input_path, output_dir, gcps_file, model_path)
```

### 사례 2: 특정 단계만 실행

경계 추출까지만 실행하고, 신축량 보정은 건너뛰려면:

```python
from src.preprocessing import image_preprocessing
from src.segmentation import multiresolution
from src.boundary_detection import svm_classifier

# 이미지 로드 및 전처리
processed_image = image_preprocessing.preprocess("data/sample/raw_cadastral_maps/map001.tif")

# 세그먼테이션 수행
segmenter = multiresolution.MultiResolutionSegmentation(scale=75, shape_weight=0.3, compactness_weight=0.5)
segment_map, segments = segmenter.segment(processed_image)

# 경계 추출
boundary_mask = segmenter.extract_boundary_mask(threshold=0.7)

# 결과 저장
import cv2
cv2.imwrite("data/results/boundaries/map001_boundaries.png", boundary_mask)
```

## 6. GCP 파일 형식

GCP(지상기준점) 파일은 다음과 같은 CSV 형식을 사용합니다:

```
mapX,mapY,pixelX,pixelY,enable,dX,dY,residual
423000.5,634200.2,100,150,1,0.0,0.0,0.0
423150.8,634350.6,250,300,1,0.0,0.0,0.0
423300.2,634250.1,400,200,1,0.0,0.0,0.0
423050.6,634450.3,150,400,1,0.0,0.0,0.0
```

각 열의 의미:
- `mapX, mapY`: 참조 좌표계의 X, Y 좌표
- `pixelX, pixelY`: 원본 이미지의 픽셀 좌표
- `enable`: 해당 GCP의 활성화 여부 (1: 활성화, 0: 비활성화)
- `dX, dY`: X, Y 방향의 오차 (초기값은 0)
- `residual`: 잔차 (초기값은 0)

## 7. 문제 해결

### 일반적인 문제

**Q: 이미지 로드 오류가 발생합니다.**

A: 이미지 파일 경로가 올바른지, 지원되는 형식인지 확인하세요. 지원되는 형식은 TIF, JPG, PNG 등입니다.

**Q: 세그먼테이션 결과가 너무 조밀하거나 너무 큽니다.**

A: `scale` 매개변수를 조정하세요. 값이 클수록 더 큰 세그먼트가 생성됩니다.

**Q: 경계 추출 결과가 만족스럽지 않습니다.**

A: 다음 방법을 시도해 보세요:
- 다른 전처리 매개변수 사용
- SVM 모델의 커널 유형이나 매개변수 조정
- 세그먼테이션 매개변수 조정

**Q: 신축량 보정 후 이미지가 왜곡됩니다.**

A: GCP 설정을 확인하세요. 최소한 3개 이상의 잘 분포된 GCP가 필요합니다. 필요한 경우 더 많은 GCP를 추가하거나 'projective' 변환 방법을 시도해 보세요.

### 기술적 문제

**Q: GDAL 관련 오류가 발생합니다.**

A: GDAL이 올바르게 설치되었는지 확인하세요. 일부 시스템에서는 GDAL을 별도로 설치해야 할 수 있습니다.

```bash
# Ubuntu
sudo apt-get install libgdal-dev
pip install gdal==$(gdal-config --version)

# macOS (Homebrew 사용)
brew install gdal
pip install gdal==$(gdal-config --version)
```

**Q: 메모리 부족 오류가 발생합니다.**

A: 대용량 이미지를 처리할 때 메모리 부족 문제가 발생할 수 있습니다. 다음 방법을 시도해 보세요:
- 이미지 크기 축소
- 작은 세그먼트 크기(scale 값 감소) 사용
- 더 많은 RAM이 있는 시스템에서 실행

**Q: 로그 파일은 어디에 저장되나요?**

A: 로그 파일은 프로젝트 루트 디렉토리의 `cadastral_processing.log`에 저장됩니다. 이 파일에서 자세한 처리 정보를 확인할 수 있습니다.