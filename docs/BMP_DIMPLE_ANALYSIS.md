
# BMP 딤플 분석 시스템

## 개요

BMP 파일의 선명도를 유지하면서 골프공의 딤플을 직접 검출하여 스핀을 분석하는 시스템입니다.

## 주요 특징

- **선명도 유지**: BMP 파일의 원본 품질을 그대로 유지
- **딤플 검출**: 고주파 필터와 특징점 검출로 딤플 인식
- **회전 추적**: 딤플의 움직임으로 정확한 스핀 측정
- **메모리 효율**: 필요한 부분만 처리하여 메모리 사용량 최적화

## 파일 구조

```
src/
├── utils/
│   └── bmp_loader.py              # BMP 전용 로더
├── processing/
│   ├── format_converter.py        # BMP 처리 도구 (변환 + 분석)
│   └── image_enhancement/
│       └── dimple_enhancer.py     # 딤플 검출 전용 전처리
├── algorithms/
│   └── spin_analysis/
│       └── no_dimple_spin_final.py # 딤플 분석 통합
└── scripts/
    └── run_bmp_analysis.py        # BMP 분석 실행 스크립트
```

## 사용 방법

### 1. BMP 직접 분석

```bash
# 기본 BMP 파일 분석
python scripts/run_bmp_analysis.py --input data/images/shot-image-original/driver/no_marker_ball-1 --club driver

# 개선된 가시성 모드 (JPG 처리 방식 적용)
python scripts/run_bmp_analysis.py --input data/images/shot-image-original/driver/no_marker_ball-1 --club driver --enhanced

# 디버그 모드
python scripts/run_bmp_analysis.py --input data/images/shot-image-original --club driver --debug --max-files 5 --enhanced
```

### 2. 포맷 변환기 사용

```bash
# JPG 변환 (기존 방식)
python src/processing/format_converter.py --mode convert --source shot-image --target shot-image-jpg

# BMP 직접 분석
python src/processing/format_converter.py --mode analyze --source data/images/shot-image-original --club driver

# 개선된 가시성 모드
python src/processing/format_converter.py --mode analyze --source data/images/shot-image-original --club driver --enhanced
```

### 3. 프로그래밍 방식

```python
from src.algorithms.spin_analysis.no_dimple_spin_final import FinalNoDimpleSpinAnalyzer

# 분석기 초기화 (BMP 분석 활성화)
analyzer = FinalNoDimpleSpinAnalyzer(enable_bmp_analysis=True)

# BMP 파일 목록
bmp_files = [
    "data/images/shot-image-original/driver/no_marker_ball-1/1_5.bmp",
    "data/images/shot-image-original/driver/no_marker_ball-1/1_6.bmp",
    "data/images/shot-image-original/driver/no_marker_ball-1/1_7.bmp"
]

# 분석 실행
result = analyzer.analyze_bmp_sequence(bmp_files, 'driver')

print(f"Total Spin: {result['total_spin']} rpm")
print(f"Backspin: {result['backspin']} rpm")
print(f"Sidespin: {result['sidespin']} rpm")
print(f"신뢰도: {result['confidence']:.1%}")
```

## 알고리즘 상세

### 1. BMP 로더 (`bmp_loader.py`)

- **PIL 기반 로딩**: 원본 품질 유지
- **메모리 캐시**: 성능 최적화
- **색상 모드 처리**: L, RGB, RGBA 자동 변환

### 2. 딤플 검출기 (`dimple_enhancer.py`)

- **고주파 강조 필터**: 딤플 특징 강화
- **HoughCircles**: 원형 딤플 검출
- **SIFT 특징점**: 미세한 딤플 검출
- **품질 평가**: 대비, 원형도, 그라데이션 분석

### 3. 딤플 추적기

- **매칭 알고리즘**: 거리 + 크기 기반 매칭
- **회전 각도 계산**: 상대 위치 기반 각도 측정
- **RPM 변환**: 820fps 기준 스핀율 계산

## 성능 최적화

### 메모리 관리
- **캐시 시스템**: 로드된 이미지 재사용
- **지연 로딩**: 필요한 시점에만 로드
- **자동 정리**: 메모리 사용량 모니터링

### 처리 속도
- **병렬 처리**: 다중 프레임 동시 분석
- **ROI 제한**: 볼 영역만 집중 분석
- **조기 종료**: 충분한 데이터 확보시 중단

## 결과 해석

### 출력 데이터
```json
{
  "total_spin": 2500,           // 총 스핀 (RPM)
  "backspin": 2125,             // 백스핀 (RPM)
  "sidespin": 375,              // 사이드스핀 (RPM)
  "spin_axis": [0.12, 0.92, 0.37], // 스핀축 벡터
  "confidence": 0.85,           // 신뢰도 (0-1)
  "method": "bmp_dimple_enhanced", // 분석 방법
  "dimple_analysis_used": true, // 딤플 분석 사용 여부
  "bmp_files_processed": 5      // 처리된 파일 수
}
```

### 신뢰도 기준
- **0.9 이상**: 매우 높음 (딤플 10개 이상 검출)
- **0.7-0.9**: 높음 (딤플 5-9개 검출)
- **0.5-0.7**: 보통 (딤플 3-4개 검출)
- **0.5 미만**: 낮음 (딤플 3개 미만 또는 검출 실패)

## 문제 해결

### 일반적인 문제

1. **딤플 검출 실패**
   - 조명 조건 확인
   - 볼 영역 정확도 확인
   - 딤플 파라미터 조정

2. **메모리 부족**
   - 캐시 비활성화: `create_bmp_loader(enable_cache=False)`
   - 파일 수 제한: `--max-files` 옵션 사용

3. **처리 속도 저하**
   - ROI 크기 조정
   - 딤플 검출 파라미터 최적화

### 디버깅

```bash
# 상세 로그 확인
python scripts/run_bmp_analysis.py --input data/images/shot-image-original --debug

# 로그 파일 확인
tail -f bmp_analysis.log
```

## 향후 개선 계획

1. **딥러닝 기반 딤플 검출**
2. **실시간 처리 최적화**
3. **다중 해상도 지원**
4. **클라우드 분석 서비스**

## 참고 자료

- [OpenCV 딤플 검출 가이드](https://docs.opencv.org/)
- [PIL 이미지 처리 문서](https://pillow.readthedocs.io/)
- [골프공 딤플 물리학](https://www.golf.com/)
