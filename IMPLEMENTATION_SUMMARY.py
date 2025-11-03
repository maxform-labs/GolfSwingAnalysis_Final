#!/usr/bin/env python3
"""
수직 스테레오 캘리브레이션 시스템 완성 요약

## 생성된 파일들:

1. **enhance_existing_calibration.py**
   - 기존 캘리브레이션에 ROI 정보와 실제 거리 보정 추가
   - 출력: precise_vertical_stereo_calibration.json

2. **golf_ball_3d_physics_analyzer.py**
   - ROI 좌표계 변환을 고려한 3D 위치 계산
   - 속력, 발사각, 방향각 계산
   - 출력: golf_ball_3d_analysis_results.json

3. **generate_analysis_report.py**
   - 계산값과 실측값 비교
   - 시각화 및 마크다운 리포트 생성
   - 출력: analysis_report.md, analysis_plots/

4. **run_complete_analysis.py**
   - 전체 파이프라인 통합 실행 스크립트

5. **PRECISE_STEREO_CALIBRATION_README.md**
   - 전체 시스템 문서화

## 주요 기능:

### 1. ROI 좌표계 변환
- 1440x1080 캘리브레이션 좌표 ↔ 1440x300 ROI 좌표
- 공식: x_full = x_roi + XOffset, y_full = y_roi + YOffset
- Cam1: YOffset=396, Cam2: YOffset=372

### 2. 실제 거리 기반 깊이 보정
- 카메라1-골프공: 900-1000mm
- 카메라2-골프공: 500-600mm
- 베이스라인: 470mm (수직 배치)
- 보정 계수: 0.1 (경험적)

### 3. 물리량 계산
- 속력: 3D 속도 벡터의 크기
- 발사각: 수평면과의 각도 (Y축이 수직)
- 방향각: X-Z 평면에서의 방향

## 현재 상태:

✓ 캘리브레이션 파일 생성 완료
✓ ROI 좌표 변환 구현 완료
✓ 골프공 검출 알고리즘 적용
✓ 3D 위치 계산 구현
✓ 물리량 계산 구현
✓ 문서화 완료

## 개선이 필요한 부분:

1. **깊이 보정 계수 정밀화**
   - 현재: 경험적 보정 계수 0.1 사용
   - 개선: 더 많은 실측 데이터로 보정 계수 최적화

2. **골프공 검출률 향상**
   - 현재: 일부 프레임에서만 검출 성공
   - 개선: 검출 파라미터 튜닝 또는 딥러닝 기반 검출기 적용

3. **속도 계산 정확도**
   - 현재: 계산된 속도가 실측값보다 2-3배 높음
   - 원인: 깊이 계산 오차, 프레임 간 매칭 문제
   - 개선: 칼만 필터 적용, 다중 프레임 융합

4. **시차 방향 결정**
   - 수직 스테레오에서 Y 방향 시차가 주가 되어야 하나
   - 현재 X 방향 시차도 큰 경우 발생
   - 카메라 정렬 상태 재확인 필요

## 사용 방법:

```bash
# 1. 캘리브레이션 개선
python enhance_existing_calibration.py

# 2. 골프공 3D 분석
python golf_ball_3d_physics_analyzer.py

# 3. 리포트 생성
python generate_analysis_report.py

# 또는 통합 실행
python run_complete_analysis.py
```

## 기술적 배경:

### 스테레오 비전 기본 원리
```
Z = (f × B) / d
```
- Z: 깊이
- f: 초점 거리 (1500 pixels)
- B: 베이스라인 (470 mm)
- d: 시차 (pixels)

### 좌표 변환
```
X = (u - cx) × Z / f
Y = (v - cy) × Z / f
```

### 속도 계산
```
v = Δposition / Δt
Δt = 1/820 ≈ 1.22 ms
```

## 결론:

기본 프레임워크는 완성되었으며, ROI 좌표 변환과 실제 거리 기반 보정이 적용되었습니다.
더 정확한 결과를 위해서는:
1. 더 많은 실측 데이터 수집
2. 보정 계수 최적화
3. 골프공 검출률 개선
4. 칼만 필터 등 고급 필터링 기법 적용

이 필요합니다.
"""
print(__doc__)
