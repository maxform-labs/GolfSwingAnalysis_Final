# 골프공 3D 분석 시스템 (Precise Vertical Stereo Vision)

수직 스테레오비전 캘리브레이션과 ROI 좌표계 변환을 통한 정밀한 골프공 3D 궤적 추적 및 물리량 분석 시스템

## 시스템 개요

### 문제 정의

고속 카메라(820fps)로 골프공을 촬영할 때 다음과 같은 기술적 제약이 있습니다:

1. **캘리브레이션**: 1440x1080 전체 이미지에서 수행
2. **실제 측정**: 1440x300 ROI(관심 영역)에서 수행
3. **좌표계 불일치**: ROI와 캘리브레이션 좌표계 간의 변환 필요
4. **깊이 계산 오차**: 스테레오 매칭 시 20-40배의 속도 오차 발생

### 해결 방안

#### 1. 정밀 수직 스테레오 캘리브레이션
- 1440x1080 전체 이미지에서 체스보드 패턴 캘리브레이션
- 렌즈 왜곡, 초점 거리, 주점 등 내부 파라미터 정확히 추출
- 카메라 간 회전 행렬(R)과 변위 벡터(T) 계산

#### 2. ROI 좌표계 변환
- ROI 좌표 → 캘리브레이션 좌표 변환 공식 적용
  ```
  x_full = x_roi + XOffset
  y_full = y_roi + YOffset
  ```
- 각 카메라의 ROI 오프셋 정보:
  - Cam1: YOffset=396
  - Cam2: YOffset=372

#### 3. 실제 거리 기반 깊이 보정
- 실측 거리 정보:
  - 카메라1 → 골프공: 900-1000mm
  - 카메라2 → 골프공: 500-600mm
  - 카메라 간 거리: 470mm (수직 배치)
- 보정 계수를 통한 깊이 스케일 조정

## 구성 요소

### 1. 캘리브레이션 모듈 (`precise_vertical_stereo_calibration.py`)

**기능:**
- 체스보드 패턴 검출 및 코너 추출
- 단일 카메라 캘리브레이션 (내부 파라미터)
- 스테레오 캘리브레이션 (외부 파라미터)
- 정류 맵(Rectification Map) 계산
- 실제 거리 기반 보정 계수 계산

**출력:**
- `precise_vertical_stereo_calibration.json`
  - 카메라 행렬 (K1, K2)
  - 왜곡 계수 (D1, D2)
  - 회전 행렬 (R), 변위 벡터 (T)
  - 투영 행렬 (P1, P2)
  - Q 행렬 (시차→깊이 변환)
  - ROI 정보
  - 깊이 보정 계수

**사용법:**
```bash
python precise_vertical_stereo_calibration.py
```

### 2. 3D 물리량 분석 모듈 (`golf_ball_3d_physics_analyzer.py`)

**기능:**
- 골프공 자동 검출 (적응형 임계값)
- ROI 좌표 → 캘리브레이션 좌표 변환
- 스테레오 매칭 및 3D 위치 계산
- 속도 벡터 계산
- 물리량 산출:
  - **속력** (m/s, km/h)
  - **발사각** (Launch Angle) - 수평면과의 각도
  - **방향각** (Direction Angle) - 수평면에서의 방향

**좌표계:**
- X축: 수평 방향
- Y축: 수직 방향 (높이)
- Z축: 깊이 (카메라로부터의 거리)

**출력:**
- `golf_ball_3d_analysis_results.json`
  - 각 샷별 검출 정보
  - 3D 위치 시퀀스
  - 속도, 발사각, 방향각

**사용법:**
```bash
python golf_ball_3d_physics_analyzer.py
```

### 3. 리포트 생성 모듈 (`generate_analysis_report.py`)

**기능:**
- 계산값과 실측값 비교
- 오차 분석 (평균, 표준편차)
- 시각화:
  - 속력 비교 그래프
  - 발사각 비교 그래프
  - 3D 궤적 시각화
- Markdown 리포트 생성

**출력:**
- `analysis_report.md` - 종합 분석 리포트
- `analysis_plots/` - 시각화 그래프들

**사용법:**
```bash
python generate_analysis_report.py
```

### 4. 통합 실행 스크립트 (`run_complete_analysis.py`)

전체 분석 파이프라인을 순차적으로 실행합니다.

**사용법:**
```bash
python run_complete_analysis.py
```

## 설치 및 실행

### 1. 필수 라이브러리 설치

```bash
pip install opencv-python numpy matplotlib
```

### 2. 데이터 구조 확인

```
GolfSwingAnalysis_Final/
├── data2/
│   ├── Calibration_image_1025/
│   │   ├── Cam1_1.bmp
│   │   ├── Cam1_2.bmp
│   │   ├── ...
│   │   ├── Cam2_1.bmp
│   │   └── Cam2_2.bmp
│   └── driver/
│       ├── 1/
│       │   ├── Cam1_*.bmp
│       │   ├── Cam2_*.bmp
│       │   ├── roi_cam1.json
│       │   └── roi_cam2.json
│       ├── 2/
│       └── ...
```

### 3. 전체 분석 실행

```bash
# 통합 스크립트 실행 (권장)
python run_complete_analysis.py

# 또는 개별 실행
python precise_vertical_stereo_calibration.py
python golf_ball_3d_physics_analyzer.py
python generate_analysis_report.py
```

## 기술적 세부사항

### 스테레오 깊이 계산

```
Z = (f × B) / d
```

- Z: 깊이 (mm)
- f: 초점 거리 (pixels)
- B: 베이스라인 (mm) - 카메라 간 거리
- d: 시차 (pixels)

### 3D 좌표 복원

```
X = (u - cx) × Z / f
Y = (v - cy) × Z / f
Z = depth
```

- (u, v): 이미지 좌표
- (cx, cy): 주점
- f: 초점 거리

### 속도 계산

```
v = Δposition / Δt
```

- Δt = frame_interval = 1/820 ≈ 1.22ms

### 발사각 계산

```
Launch Angle = arctan(vy / sqrt(vx² + vz²))
```

### 방향각 계산

```
Direction Angle = arctan2(vx, vz)
```

## 캘리브레이션 품질 지표

### 재투영 오차 (Reprojection Error)
- **목표**: < 0.5 pixels
- 체스보드 코너의 실제 위치와 재투영 위치 간의 차이

### 스테레오 오차 (Stereo Error)
- **목표**: < 1.0 pixels
- 스테레오 캘리브레이션의 전체 RMS 오차

### 베이스라인 오차
- **목표**: < 5mm
- 계산된 베이스라인과 실측 베이스라인(470mm)의 차이

## 결과 해석

### 속도 오차
- **허용 범위**: ±5%
- 실측 속도와 계산 속도의 백분율 차이

### 발사각 오차
- **허용 범위**: ±2°
- 실측 발사각과 계산 발사각의 차이

### 방향각 오차
- **허용 범위**: ±3°
- 실측 방향각과 계산 방향각의 차이

## 문제 해결

### 1. 캘리브레이션 실패
- 체스보드가 잘 보이는 이미지인지 확인
- 최소 10장 이상의 다양한 각도 이미지 필요
- 조명 균일성 확인

### 2. 골프공 검출 실패
- 배경과 골프공의 대비 확인
- ROI 영역이 올바르게 설정되었는지 확인
- 적응형 임계값 파라미터 조정

### 3. 깊이 계산 오차
- 실제 거리 측정값 재확인
- 캘리브레이션 재수행
- 보정 계수 조정

### 4. 속도 오차가 큰 경우
- 프레임 레이트 확인 (820fps)
- 연속 프레임에서 골프공 검출 성공률 확인
- 더 많은 프레임 데이터 사용

## 향후 개선 방향

1. **딥러닝 기반 골프공 검출**
   - YOLO, Faster R-CNN 등 적용
   - 낮은 조명에서도 강건한 검출

2. **칼만 필터 적용**
   - 다중 프레임 데이터 융합
   - 노이즈 제거 및 궤적 평활화

3. **자동 캘리브레이션**
   - 실시간 캘리브레이션 검증
   - 온라인 재캘리브레이션

4. **GPU 가속**
   - CUDA 기반 스테레오 매칭
   - 실시간 처리 성능 향상

## 참고 문헌

1. Zhang, Z. (2000). "A flexible new technique for camera calibration"
2. Hartley, R., & Zisserman, A. (2003). "Multiple View Geometry in Computer Vision"
3. OpenCV Documentation: Camera Calibration and 3D Reconstruction

## 라이선스

MIT License

## 저자

Golf Swing Analysis Project Team

## 문의

프로젝트 관련 문의사항이나 버그 리포트는 이슈 트래커를 이용해주세요.

---

**마지막 업데이트:** 2025-10-30
