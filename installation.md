# 지적원도 객체분석 및 신축량 보정 자동화 시스템 설치 가이드

본 문서는 지적원도 객체분석 및 신축량 보정 자동화 시스템의 설치 방법을 상세히 설명합니다.

## 목차

1. [시스템 요구사항](#1-시스템-요구사항)
2. [기본 설치](#2-기본-설치)
3. [운영체제별 설치 가이드](#3-운영체제별-설치-가이드)
4. [GDAL 설치](#4-gdal-설치)
5. [추가 라이브러리 설치](#5-추가-라이브러리-설치)
6. [설치 확인](#6-설치-확인)
7. [문제 해결](#7-문제-해결)

## 1. 시스템 요구사항

### 하드웨어 요구사항
- **CPU**: 멀티코어 프로세서 (Intel Core i5 이상 또는 동급 사양)
- **RAM**: 최소 8GB, 권장 16GB 이상 (대용량 이미지 처리 시 더 많은 메모리 필요)
- **디스크 공간**: 최소 5GB (샘플 데이터 및 결과 저장용)
- **그래픽 카드**: 일반적인 처리에는 필수 아님 (향후 딥러닝 모델 지원 시 필요할 수 있음)

### 소프트웨어 요구사항
- **운영체제**: Windows 10/11, macOS 10.15 이상, Ubuntu 20.04 이상
- **Python**: 3.8 이상
- **필수 라이브러리**:
  - GDAL 3.4.0 이상
  - OpenCV 4.5.3 이상
  - NumPy 1.20.0 이상
  - scikit-learn 1.0.0 이상
  - scikit-image 0.18.0 이상
  - matplotlib 3.4.0 이상
  - 기타: 전체 목록은 requirements.txt 참조

## 2. 기본 설치

### 저장소 복제
```bash
git clone https://github.com/yourusername/cadastral-map-automation.git
cd cadastral-map-automation
```

### 가상 환경 생성 및 활성화
Python 가상 환경을 사용하여 의존성 충돌을 방지하는 것이 좋습니다.

```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 기본 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

## 3. 운영체제별 설치 가이드

### Windows

#### 사전 준비
1. [Python 3.8 이상](https://www.python.org/downloads/windows/) 설치
2. [Git for Windows](https://gitforwindows.org/) 설치

#### GDAL 설치
Windows에서는 GDAL의 바이너리 버전을 설치하는 것이 좋습니다.

1. [Christoph Gohlke의 비공식 Python 확장 패키지](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)에서 시스템에 맞는 GDAL 휠 파일 다운로드
   - 파일명 예시: `GDAL-3.4.3-cp38-cp38-win_amd64.whl` (Python 3.8, 64비트 Windows용)

2. 다운로드한 휠 파일 설치
```bash
pip install C:\path\to\downloaded\GDAL-3.4.3-cp38-cp38-win_amd64.whl
```

#### 추가 종속성 설치
```bash
pip install requirements.txt
```

### macOS

#### Homebrew를 이용한 설치
1. [Homebrew](https://brew.sh/) 설치 (아직 설치하지 않은 경우)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Python 및 GDAL 설치
```bash
brew install python gdal
```

3. 가상 환경 생성 및 의존성 설치
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

1. 시스템 패키지 설치
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev \
    libgdal-dev gdal-bin libspatialindex-dev \
    libopencv-dev
```

2. GDAL 버전 확인 및 Python 바인딩 설치
```bash
gdal-config --version
pip install gdal==$(gdal-config --version)
```

3. 가상 환경 생성 및 의존성 설치
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. GDAL 설치

GDAL(Geospatial Data Abstraction Library)은 지리공간 데이터 처리를 위한 필수 라이브러리입니다. 운영체제별로 설치 방법이 다릅니다.

### GDAL 환경 변수 설정 (Windows)

Windows에서 GDAL을 설치한 후에는 환경 변수를 설정해야 할 수 있습니다.

1. 시스템 환경 변수 편집기 열기
2. 다음 환경 변수 추가:
   - `GDAL_DATA`: `C:\path\to\venv\Lib\site-packages\osgeo\data\gdal`
   - `PROJ_LIB`: `C:\path\to\venv\Lib\site-packages\osgeo\data\proj`

### GDAL 설치 확인

설치가 제대로 되었는지 확인하려면 다음 명령을 실행합니다:

```bash
python -c "from osgeo import gdal; print(gdal.__version__)"
```

버전 번호가 출력되면 GDAL이 올바르게 설치된 것입니다.

## 5. 추가 라이브러리 설치

### OpenCV 최적화 (선택 사항)

더 나은 성능을 위해 OpenCV 최적화 버전을 설치할 수 있습니다:

```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

### GPU 가속 (향후 지원)

향후 딥러닝 모델을 지원할 경우, NVIDIA GPU가 있다면 CUDA 및 cuDNN을 설치하여 처리 속도를 향상시킬 수 있습니다.

## 6. 설치 확인

설치가 제대로 되었는지 확인하기 위해 테스트 스크립트를 실행합니다:

```bash
python src/test_installation.py
```

이 스크립트는 필요한 모든 라이브러리를 가져오고 기본 기능을 테스트합니다. 오류 없이 실행되면 시스템이 올바르게 설정된 것입니다.

## 7. 문제 해결

### 일반적인 설치 문제

#### GDAL 설치 오류

**문제**: `ERROR: Could not build wheels for GDAL which use PEP 517...`

**해결책**:
1. 시스템 수준에서 GDAL을 먼저 설치합니다.
2. 그 다음 Python 바인딩을 설치합니다.
```bash
# Ubuntu
sudo apt-get install libgdal-dev
pip install gdal==$(gdal-config --version)

# macOS
brew install gdal
pip install gdal==$(gdal-config --version)
```

#### OpenCV 관련 오류

**문제**: `ImportError: libGL.so.1: cannot open shared object file...`

**해결책** (Ubuntu):
```bash
sudo apt-get install libgl1-mesa-glx
```

#### 메모리 오류

**문제**: 큰 이미지 처리 시 `MemoryError` 발생

**해결책**:
1. 더 많은 RAM이 있는 시스템 사용
2. 이미지 크기 조정 옵션 활성화
3. 시스템의 가상 메모리/스왑 공간 증가

### 운영체제별 문제해결

#### Windows

**문제**: `OSError: [WinError 126] 지정된 모듈을 찾을 수 없습니다.`

**해결책**:
1. GDAL DLL이 경로에 있는지 확인합니다.
2. Python 설치 디렉토리의 DLLs 폴더에 필요한 GDAL DLL을 복사합니다.

#### macOS

**문제**: `ImportError: dlopen(...): Library not loaded: @loader_path/...`

**해결책**:
```bash
brew install gcc@9
export CFLAGS="-I/usr/local/include"
export LDFLAGS="-L/usr/local/lib"
pip install --no-binary :all: gdal
```

#### Linux

**문제**: `ImportError: libopencv_core.so.X.X: cannot open shared object file...`

**해결책**:
```bash
# OpenCV 의존성 패키지 설치
sudo apt-get install libopencv-dev
# 또는 심볼릭 링크 생성
sudo ln -s /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2 /usr/lib/x86_64-linux-gnu/libopencv_core.so.X.X
```

### 기타 문제

도움이 필요하시면 GitHub 이슈 페이지에 문제를 보고해 주세요. 다음 정보를 포함하면 도움이 됩니다:
- 운영체제 및 버전
- Python 버전
- 설치 과정
- 오류 메시지 (전체 오류 로그)
- 시도해 본 해결책