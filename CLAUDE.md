# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

**골프 스윙 분석 시스템** - 수직 스테레오 비전을 사용한 실시간 240fps 스윙 분석 시스템입니다. 상용 키오스크 배포를 위해 설계되었으며 94% 이상의 높은 정확도를 목표로 합니다.

## 주요 명령어

### 개발 환경 설정
```bash
# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 의존성 설치
pip install -r GolfSwingAnalysis_Final_ver8/requirements.txt
```

### 시스템 실행
```bash
# 메인 분석 시스템 (240fps 실시간 처리)
cd GolfSwingAnalysis_Final_ver8
python golf_swing_analyzer.py

# 키오스크 GUI 시스템 (터치스크린 인터페이스)
python kiosk_system.py

# 웹 대시보드 (Flask 서버, 포트 5000)
python web_dashboard.py

# 시뮬레이션 환경 (하드웨어 없이 테스트)
python simulation_environment.py

# 검증 시스템
python realistic_achievement_system.py  # 95% 정확도 검증
python realistic_3d_validation_app.py   # 3D 시각화 (포트 5002)
```

### 테스트
```bash
# 검증 테스트 실행
python test_corrections.py

# 통합 테스트
python test_integration.py

# Playwright 테스트 (필요시)
python test_playwright.py
```

## 시스템 아키텍처

### 핵심 처리 파이프라인
실시간 240fps 처리를 위한 **4개 스레드 병렬 파이프라인**:

1. **프레임 캡처 스레드** (1ms): 상/하 카메라 동기화 캡처
2. **객체 검출 스레드** (5ms): ROI 추적을 통한 공/클럽 검출
3. **스테레오 분석 스레드** (8ms): Y축 시차 계산, 3D 좌표 변환, 칼만 필터링
4. **결과 출력 스레드** (2ms): 데이터 검증, DB 저장, UI 업데이트

총 레이턴시: 프레임당 16ms 이하

### 수직 스테레오 비전 구성
- **베이스라인**: 카메라 간 400mm 수직 간격
- **내향 각도**: 12° 최적 수렴각
- **핵심 공식**: Y축 시차용 `Z = (fy × baseline) / (y1 - y2)`

### 주요 알고리즘 구성요소

#### 고급 칼만 필터 (`advanced_algorithms.py`)
- 6상태 필터: [x, y, z, vx, vy, vz]
- 측정 품질 기반 적응형 노이즈 튜닝
- 프로세스 노이즈: Q = 0.01 (위치), 0.1 (속도)
- 측정 노이즈: R = 0.1 (위치), 0.5 (각도)

#### 베이지안 앙상블 (구현 필요)
**중요**: 현재 3개 추정기 중 1개만 구현됨. 추가 필요:
- `ParticleFilterEstimator`
- `LeastSquaresEstimator`

#### 적응형 보정 (`adaptive_correction.py`)
- 스킬 레벨별 보정 (초급/중급/상급/프로)
- 물리적 제약 검증
- 신뢰도 가중 조정

## 알려진 문제 및 검증 결과

`golf_swing_validation_report.json` 및 `corrected_algorithm_specification.md` 기반:

### 심각한 문제
1. **Y축 시차**: 구현이 명시적으로 Y축에 집중하지 않음
2. **베이지안 앙상블**: 필수 3개 추정기 중 2개 누락
3. **칼만 필터 불일치**: 6상태와 가변 상태 필터 혼용

### 정확도 목표
- 볼 스피드: ±3.5%
- 발사각: ±2.5%
- 방향각: ±4%
- 클럽 스피드: ±3.5%
- 어택 앵글: ±4.5%
- 클럽 패스: ±4%
- 페이스 앵글: ±7%

## 설정 파일

### `config.json`
카메라 설정, 스테레오 파라미터, UI 설정을 위한 메인 구성. 주요 파라미터:
- 카메라 FPS: 60 (주의: 시스템은 240fps용으로 설계됨, 조정 필요할 수 있음)
- 수직 베이스라인: 400mm
- 내향 각도: 12°
- 해상도: 1280x720

## 데이터 흐름

```
[카메라 입력] → [IR 동기화] → [프레임 캡처] → [전처리]
                                    ↓
[UI 업데이트] ← [DB 저장] ← [결과 검증] ← [물리 계산]
                                    ↑
                            [3D 좌표] ← [스테레오 매칭] ← [객체 검출]
```

## 성능 최적화 기능

- **메모리 풀**: 효율성을 위한 프레임 버퍼 재사용
- **GPU 가속**: OpenCV를 통한 CUDA 지원 (선택적)
- **적응형 ROI**: 처리 속도를 위한 동적 관심 영역
- **큐 관리**: 제한된 큐를 사용한 비동기 처리

## API 엔드포인트 (웹 대시보드)

- `/api/status` - 시스템 상태
- `/api/start` - 분석 시작
- `/api/stop` - 분석 중지
- `/api/results` - 분석 결과 조회
- `/api/calibrate` - 카메라 캘리브레이션

## 개발 참고사항

### 멀티스레딩 고려사항
시스템은 4개 워커로 `ThreadPoolExecutor`를 사용합니다. 각 처리 단계는 큐 기반 통신으로 자체 스레드에서 실행됩니다. 공유 상태에 주의하고 적절한 동기화를 사용하세요.

### IR 동기화
IR 조명 시스템 (`ir_synchronization.py`)은 240fps에서 카메라 캡처와 동기화되어야 합니다. 시스템은 자동 샷 감지와 적응형 강도 제어를 지원합니다.

### 좌표계
- 수직 스테레오 구성 사용 (카메라가 수직으로 배치)
- Y축 시차가 주요 깊이 단서 (수평 스테레오의 X축이 아님)
- 12° 내향 각도는 회전 행렬 보정 필요

### 주요 용어
- 수직 스테레오 비전 (Vertical Stereo Vision)
- 칼만 필터 (Kalman Filter)
- 베이지안 추정기 (Bayesian Estimator)
- 적응형 보정 (Adaptive Correction)
- 어택 앵글 (Attack Angle)
- 클럽 패스 (Club Path)
- 페이스 앵글 (Face Angle)