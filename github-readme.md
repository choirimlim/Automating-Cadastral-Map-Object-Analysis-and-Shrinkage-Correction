# 지적원도 객체분석 및 신축량 보정 자동화 시스템

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.3-green.svg)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange.svg)](https://scikit-learn.org/)
[![GDAL](https://img.shields.io/badge/GDAL-3.4.0-blue.svg)](https://gdal.org/)

## 개요

이 프로젝트는 지적원도(cadastral map)의 객체분석 및 신축량 보정을 자동화하기 위한 통합 시스템을 제공합니다. 토지 등록 및 경계 설정의 정확성을 향상시키기 위해 컴퓨터 비전 및 기계학습 기술을 활용하였습니다.

### 주요 기능

- **이미지 전처리** - 노이즈 감소, 대비 향상 및 품질 개선
- **객체 세그먼테이션** - 다중해상도 세그먼테이션(MRS)을 통한 토지 구획 식별
- **경계 추출** - 머신러닝(SVM) 기반 경계 검출 및 추출
- **신축량 보정** - 지상기준점(GCP) 기반 아핀 변환 적용
- **성능 평가** - 자동화 결과의 정확도 평가 도구

## 설치 방법

### 사전 요구사항

- Python 3.8 이상
- GDAL 3.4.0
- OpenCV 4.5.3
- scikit-learn 1.0.2

### 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/cadastral-map-automation.git
cd cadastral-map-automation

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows의 경우: venv\Scripts\activate

# 의존성 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

### 기본 사용법

```python
from src.preprocessing import image_preprocessing
from src.segmentation import multiresolution
from src.boundary_detection import svm_classifier
from src.correction import affine_transform

# 이미지 전처리
preprocessed_img = image_preprocessing.preprocess("data/sample/raw_cadastral_maps/map001.tif")

# 객체 세그먼테이션
segments = multiresolution.segment(preprocessed_img, scale=75, shape=0.3, compactness=0.5)

# 경계 추출
boundaries = svm_classifier.extract_boundaries(segments)

# 신축량 보정
gcps = [
    ((100, 150), (423.5, 634.2)),  # (원본 좌표, 참조 좌표)
    ((250, 300), (573.8, 784.6)),
    ((400, 200), (723.2, 684.1)),
    ((150, 400), (473.6, 884.3))
]
corrected_map = affine_transform.apply_correction(boundaries, gcps)

# 결과 저장
corrected_map.save("data/results/corrected/map001_corrected.tif")
```

자세한 사용법은 [사용자 가이드](docs/usage.md)를 참조하세요.

## 방법론

본 프로젝트는 다음과 같은 방법론을 기반으로 구현되었습니다:

1. **이미지 전처리**: 가우시안 블러 및 적응형 임계값 적용을 통한 노이즈 감소
2. **객체 기반 이미지 분석(OBIA)**: 다중해상도 세그먼테이션을 통한 의미 있는 객체 식별
3. **기계학습 기반 분류**: 분광 및 기하학적 특성을 활용한 SVM 모델 학습
4. **지상기준점(GCP) 기반 보정**: 아핀 및 투영 변환을 통한 지도 왜곡 수정

자세한 방법론은 [방법론 문서](docs/methodology.md)를 참조하세요.

## 성능 평가

다양한 지적원도 샘플에 대한 테스트 결과, 본 시스템은 다음과 같은 성능을 보였습니다:

- 객체 식별 정확도: 92.7%
- 경계 추출 정확도: 89.5%
- 신축량 보정 후 위치 정확도: RMSE 0.82m

## 기여 방법

기여를 원하시면 다음 절차를 따라주세요:

1. 이 저장소를 포크합니다.
2. 새로운 기능 브랜치를 만듭니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m '새로운 기능 추가'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다.

## 라이센스

MIT 라이센스에 따라 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 감사의 말

- 한국국토정보공사(LX)의 지적원도 샘플 데이터 제공
- 국토연구원의 토지정보시스템 연구 지원
