# Data Standard Guide (data-standard.xlsx)

**파일 위치**: `C:\src\GolfSwingAnalysis_Final\data\data-standard.xlsx`
**생성일**: 2025-10-29
**버전**: 1.0
**목적**: Phase 3 골프 스윙 분석 알고리즘 검증을 위한 Ground Truth 데이터

---

## 📋 목차

1. [개요](#개요)
2. [파일 구조](#파일-구조)
3. [각 시트 설명](#각-시트-설명)
4. [Phase 3 사용 방법](#phase-3-사용-방법)
5. [정확도 계산 공식](#정확도-계산-공식)
6. [검증 결과](#검증-결과)
7. [참고 자료](#참고-자료)

---

## 개요

### 목적
`data-standard.xlsx`는 골프 스윙 분석 알고리즘의 측정 정확도를 검증하기 위한 **Ground Truth 데이터셋**입니다.

### 핵심 기능
1. **PGA Tour 표준값**: 프로 골퍼 평균 데이터를 기준으로 제공
2. **실측 데이터 비교**: 실제 측정값과의 오차율 자동 계산
3. **Phase 3 준비**: 7번 아이언 이미지 분석 결과 입력 준비
4. **참조 문서화**: 13개 권위있는 출처의 URL 및 설명 포함

### 데이터 출처
- **TrackMan PGA Tour Averages 2024**: 프로 투어 선수 평균 데이터
- **MyGolfSpy Optimal Launch Chart**: 최적 발사 조건 범위
- **Golf.com**: 드라이버 최적 스핀율 및 발사각
- **PING WRX**: 클럽별 측정 기준값
- **실측 데이터**: `shotdata_20251020.csv` (드라이버 20샷)

---

## 파일 구조

### 4개 시트 구성

| 시트명 | 목적 | 데이터 상태 | Phase |
|--------|------|-------------|-------|
| **Driver_Ball_Standard** | 드라이버 볼 데이터 기준 | ✅ 실측 20샷 포함 | Phase 2 완료 |
| **Driver_Club_Standard** | 드라이버 클럽 파라미터 기준 | ⏳ 빈 컬럼 (입력 대기) | Phase 3 대기 |
| **7Iron_Standard** | 7번 아이언 기준 | ⏳ Phase3 빈 컬럼 | Phase 3 대상 |
| **Reference_URLs** | 참고 자료 출처 | ✅ 24개 출처 문서화 | - |

### 검증 통계 (recalc.py 결과)
```
- 총 시트: 4개
- 총 공식: 44개 (에러 0개)
- 총 데이터 셀: 426개
- 참조 URL: 24개
- Phase 3 준비 상태: ✅ True
```

---

## 각 시트 설명

### 1. Driver_Ball_Standard (드라이버 볼 데이터)

#### 구조
- **Row 1**: 시트 제목
- **Row 2**: 데이터 출처 설명
- **Row 4**: 컬럼 헤더
- **Row 5-16**: PGA Tour 표준값 (12개 측정값)
- **Row 17-36**: 실측 데이터 (CSV에서 20샷)
- **Row 40-43**: 통계 요약 (평균, 최소, 최대, 표준편차)

#### 컬럼 설명

| 컬럼 | 이름 | 설명 | 예시 |
|------|------|------|------|
| A | Measurement | 측정 항목명 | Ball Speed |
| B | PGA Tour Average | 프로 평균값 | 165.0 mph |
| C | Optimal Min | 최적 범위 최소값 | 155.0 mph |
| D | Optimal Max | 최적 범위 최대값 | 180.0 mph |
| E | Unit | 측정 단위 | mph, m/s, degrees, rpm |
| F | Actual (CSV) | 실측 평균값 (Excel 공식) | `=B40*2.23694` |
| G | Error (%) | 오차율 (Excel 공식) | `=IF(F5<>"",ABS((F5-B5)/B5)*100,"")` |
| H | Reference | 데이터 출처 | MyGolfSpy, Golf.com |

#### 측정 항목 (12개)

| # | 측정값 | PGA 표준 | 단위 | 출처 |
|---|--------|----------|------|------|
| 1 | Ball Speed | 165.0 | mph | MyGolfSpy, Golf.com |
| 2 | Ball Speed | 73.7 | m/s | Converted |
| 3 | Launch Angle | 12.0 | degrees | MyGolfSpy optimal |
| 4 | Total Spin | 2300.0 | rpm | Golf.com, PING |
| 5 | Back Spin | 2200.0 | rpm | TrackMan average |
| 6 | Side Spin | 300.0 | rpm | PING optimal |
| 7 | Spin Axis | 2.0 | degrees | MyGolfSpy |
| 8 | Launch Direction | 0.0 | degrees | Straight target |
| 9 | Carry Distance | 275.0 | yards | PGA Tour average |
| 10 | Total Distance | 295.0 | yards | Total roll |
| 11 | Max Height | 32.0 | yards | Optimal trajectory |
| 12 | Landing Angle | 45.0 | degrees | Optimal descent |

#### 실측 데이터 (Row 17-36)
- **DateTime**: 2025-10-20 측정 타임스탬프
- **Club**: driver
- **BallSpeed(m/s)**: 53.9 ~ 63.7 m/s
- **LaunchAngle(deg)**: 3.5 ~ 14.3 degrees
- **LaunchDirection(deg)**: -9.0 ~ 9.6 degrees
- **TotalSpin(rpm)**: 2068 ~ 5064 rpm
- **SpinAxis(deg)**: -15.3 ~ 13.5 degrees
- **BackSpin(rpm)**: 2000 ~ 4800 rpm
- **SideSpin(rpm)**: -800 ~ 650 rpm

#### 통계 요약 (Row 40-43)
- **평균**: `=AVERAGE(C17:C36)`
- **최소**: `=MIN(C17:C36)`
- **최대**: `=MAX(C17:C36)`
- **표준편차**: `=STDEV(C17:C36)`
- **카운트**: `=COUNT(C17:C36)`

---

### 2. Driver_Club_Standard (드라이버 클럽 파라미터)

#### 구조
- **Row 4**: 컬럼 헤더
- **Row 5-10**: 클럽 측정 항목 (6개)

#### 측정 항목 (6개)

| # | 측정값 | PGA 표준 | 단위 | 출처 |
|---|--------|----------|------|------|
| 1 | Club Speed | 113.0 | mph | TrackMan Tour |
| 2 | Attack Angle | 3.0 | degrees | TrackMan optimal |
| 3 | Club Path | 2.0 | degrees | PING in-to-out |
| 4 | Face Angle | 1.0 | degrees | Slight closed |
| 5 | Dynamic Loft | 14.0 | degrees | TrackMan average |
| 6 | Smash Factor | 1.46 | ratio | Tour average |

#### 상태
- **Actual 컬럼**: ⏳ 빈 상태 (Phase 3에서 입력 예정)
- **Error (%) 공식**: `=IF(F5<>"",ABS((F5-B5)/B5)*100,"")` (준비 완료)

---

### 3. 7Iron_Standard (7번 아이언)

#### 구조
- **Row 4**: 컬럼 헤더
- **Row 5-14**: 측정 항목 (10개)

#### 측정 항목 (10개)

| # | 측정값 | PGA 표준 | 단위 | 출처 |
|---|--------|----------|------|------|
| 1 | Club Speed | 90.0 | mph | TrackMan Tour |
| 2 | Ball Speed | 120.0 | mph | TrackMan Tour |
| 3 | Ball Speed | 53.6 | m/s | Converted |
| 4 | Launch Angle | 16.3 | degrees | TrackMan 16.3° |
| 5 | Spin Rate | 7097.0 | rpm | TrackMan 7097 |
| 6 | Attack Angle | -4.0 | degrees | TrackMan descending |
| 7 | Club Path | 0.5 | degrees | TrackMan slight in-to-out |
| 8 | Face Angle | 0.0 | degrees | Square to target |
| 9 | Smash Factor | 1.33 | ratio | TrackMan Tour |
| 10 | Carry Distance | 172.0 | yards | TrackMan Tour |

#### Phase 3 준비
- **Phase3 Measured 컬럼**: ⏳ 빈 상태
- **사용 방법**: Phase 3에서 이미지 분석 결과를 이 컬럼에 입력
- **자동 계산**: 입력 시 Error (%) 자동 계산됨

---

### 4. Reference_URLs (참고 자료)

#### 구조
- **Row 1**: 헤더
- **Row 2-25**: 24개 참조 출처

#### 카테고리별 출처

##### Driver 관련 (8개)
1. **MyGolfSpy - Optimal Launch Chart**
   - URL: https://mygolfspy.com/news-opinion/instruction/optimal-launch-and-spin-chart-for-drivers-are-you-in-the-right-range/
   - 내용: Ball speed, launch, spin ranges

2. **Golf.com - Driver Optimization**
   - URL: https://www.golf.com/instruction/how-to-optimize-your-driver/
   - 내용: 12° launch, 2300 rpm spin

3. **PING WRX - Driver Specs**
   - URL: https://www.ping.com/en-us/clubs/drivers
   - 내용: Dynamic loft, smash factor

##### 7 Iron 관련 (4개)
4. **TrackMan Tour Averages 2024**
   - URL: https://www.trackman.com/blog/golf/introducing-updated-tour-averages
   - 내용: PGA/LPGA Tour statistics

5. **TrackMan PGA Stats**
   - URL: https://blog.trackmangolf.com/trackman-average-tour-stats/
   - 내용: Club 90mph, Ball 120mph, 7097rpm

##### Project 문서 (12개)
6. **After-Calibration Plan**
   - 경로: `C:\src\GolfSwingAnalysis_Final\docs\after-calibration-plan.md`
   - 내용: Phase 3 plan, baseline 470mm

7. **통합 골프스윙분석 알고리즘 최종설계서 v6.0**
   - 경로: `C:\src\GolfSwingAnalysis_Final\docs\algorithm_specs\통합_골프스윙분석_알고리즘_최종설계서_v6.0.md`
   - 내용: 820fps system design

---

## Phase 3 사용 방법

### 워크플로우

```
1. 이미지 분석 실행
   └─> C:\src\GolfSwingAnalysis_Final\data\1440_300_data\7i\

2. 측정값 추출
   - Ball Speed (m/s)
   - Launch Angle (degrees)
   - Spin Rate (rpm)
   - Attack Angle (degrees)
   - Club Path (degrees)
   - Face Angle (degrees)
   - Smash Factor (ratio)
   - Carry Distance (yards)
   - 등 10개 항목

3. data-standard.xlsx 열기
   └─> 시트: "7Iron_Standard"

4. "Phase3 Measured" 컬럼(F열)에 측정값 입력
   - F5: Club Speed
   - F6: Ball Speed (mph)
   - F7: Ball Speed (m/s)
   - ...
   - F14: Carry Distance

5. Error (%) 자동 계산 확인
   └─> G열에 오차율이 자동으로 계산됨

6. 정확도 분석
   └─> 목표: ±3.5% 이내 (95% 정확도 달성)
```

### 입력 예시

**시나리오**: 7번 아이언 이미지 분석 완료

| Measurement | PGA Standard | Phase3 Measured | Error (%) |
|-------------|--------------|-----------------|-----------|
| Club Speed | 90.0 mph | 89.2 mph | 0.89% ✅ |
| Ball Speed | 120.0 mph | 118.5 mph | 1.25% ✅ |
| Launch Angle | 16.3 degrees | 17.1 degrees | 4.91% ⚠️ |
| Spin Rate | 7097 rpm | 7250 rpm | 2.16% ✅ |

**해석**:
- ✅ **3개 항목**: ±3.5% 이내 (우수)
- ⚠️ **1개 항목**: ±5% 이내 (개선 필요)

---

## 정확도 계산 공식

### Excel 공식 구조

#### 오차율 계산
```excel
=IF(F5<>"", ABS((F5-B5)/B5)*100, "")
```

**설명**:
- `F5<>""`: Phase3 측정값이 입력되었는지 확인
- `ABS((F5-B5)/B5)*100`: 절대 오차율 계산
  - `F5`: 실측값 (Phase3 Measured)
  - `B5`: 표준값 (PGA Tour Average)
  - `(F5-B5)/B5`: 상대 오차
  - `ABS()`: 절댓값
  - `*100`: 백분율 변환

#### 통계 계산
```excel
평균: =AVERAGE(C17:C36)
최소: =MIN(C17:C36)
최대: =MAX(C17:C36)
표준편차: =STDEV(C17:C36)
카운트: =COUNT(C17:C36)
```

#### 단위 변환
```excel
mph to m/s: =B5*0.44704
m/s to mph: =B5*2.23694
yards to meters: =B5*0.9144
```

### Python 계산 예시

```python
import pandas as pd
import openpyxl

# Excel 파일 로드
wb = openpyxl.load_workbook('data/data-standard.xlsx')
ws = wb['7Iron_Standard']

# Phase 3 측정값 입력 (예시)
phase3_measurements = {
    'F5': 89.2,   # Club Speed
    'F6': 118.5,  # Ball Speed (mph)
    'F7': 53.0,   # Ball Speed (m/s)
    'F8': 17.1,   # Launch Angle
    'F9': 7250,   # Spin Rate
}

for cell, value in phase3_measurements.items():
    ws[cell] = value

# 저장
wb.save('data/data-standard_phase3_updated.xlsx')

# 오차율 계산 (Python으로 검증)
for row in range(5, 15):
    standard = ws[f'B{row}'].value
    measured = ws[f'F{row}'].value
    if measured:
        error = abs((measured - standard) / standard) * 100
        print(f"{ws[f'A{row}'].value}: {error:.2f}%")
```

### 정확도 목표

| 측정값 카테고리 | 목표 오차율 | 등급 |
|----------------|-------------|------|
| Ball Speed | ±3.0% | 우수 |
| Launch Angle | ±2.5% | 우수 |
| Spin Rate | ±8.0% | 양호 |
| Club Speed | ±3.5% | 우수 |
| Attack Angle | ±4.5% | 양호 |
| Face Angle | ±5.0% | 양호 |

**전체 시스템 목표**: **95% 정확도** (모든 측정값이 목표 오차율 이내)

---

## 검증 결과

### recalc.py 검증 완료 (2025-10-29)

```json
{
  "status": "success",
  "file": "data\\data-standard.xlsx",
  "sheets": {
    "Driver_Ball_Standard": {
      "total_cells": 344,
      "formula_cells": 31,
      "data_cells": 238,
      "empty_cells": 75
    },
    "Driver_Club_Standard": {
      "total_cells": 104,
      "formula_cells": 4,
      "data_cells": 47,
      "empty_cells": 53
    },
    "7Iron_Standard": {
      "total_cells": 144,
      "formula_cells": 9,
      "data_cells": 72,
      "empty_cells": 63
    },
    "Reference_URLs": {
      "total_cells": 100,
      "formula_cells": 0,
      "data_cells": 69,
      "empty_cells": 31
    }
  },
  "summary": {
    "total_sheets": 4,
    "phase3_ready": true,
    "reference_count": 24,
    "total_formulas": 44,
    "total_data_cells": 426
  }
}
```

### 검증 항목

✅ **구조 검증**
- 4개 시트 모두 존재
- 필수 컬럼 헤더 확인 (Row 4)
- Driver 실측 데이터 20샷 포함

✅ **공식 검증**
- 44개 Excel 공식 모두 정상
- 에러 값 없음 (#REF!, #DIV/0!, #VALUE!, #NAME? 등)

✅ **Phase 3 준비**
- 7Iron_Standard 시트의 "Phase3 Measured" 컬럼 존재
- Error (%) 공식 준비 완료
- 자동 계산 로직 테스트 완료

✅ **참조 문서화**
- 24개 권위있는 출처 문서화
- TrackMan, MyGolfSpy, Golf.com, PING 등
- 프로젝트 내부 문서 12개 링크

---

## 참고 자료

### 공식 문서
- **After-Calibration Plan**: `docs/after-calibration-plan.md`
- **알고리즘 설계서 v6.0**: `docs/algorithm_specs/통합_골프스윙분석_알고리즘_최종설계서_v6.0.md`
- **CLAUDE.md**: 프로젝트 가이드

### 스크립트
- **create_data_standard.py**: 이 파일 생성 스크립트
- **recalc.py**: Excel 공식 검증 스크립트

### 외부 출처
1. **TrackMan**: https://www.trackman.com/blog/golf/
2. **MyGolfSpy**: https://mygolfspy.com/
3. **Golf.com**: https://www.golf.com/
4. **PING**: https://www.ping.com/

### 관련 파일
- **shotdata_20251020.csv**: 드라이버 실측 데이터 (20샷)
- **multi_club_analysis_results.json**: 7i, 5i, PW 분석 결과

---

## 다음 단계 (Phase 3)

### 준비사항
1. ✅ data-standard.xlsx 생성 완료
2. ✅ Excel 공식 검증 완료
3. ✅ 참조 문서화 완료
4. ⏳ 7번 아이언 이미지 분석 준비

### Phase 3 실행 계획
1. **이미지 로드**: `C:\src\GolfSwingAnalysis_Final\data\1440_300_data\7i\`
2. **알고리즘 실행**: 820fps 수직 스테레오 비전 분석
3. **측정값 추출**: 10개 파라미터 자동 추출
4. **결과 입력**: data-standard.xlsx의 Phase3 Measured 컬럼
5. **정확도 산출**: Error (%) 자동 계산 및 리포트 생성

### 성공 기준
- **목표**: 95% 정확도 달성
- **평가**: 모든 측정값이 목표 오차율 ±3.5% 이내
- **보고서**: Phase 3 완료 후 accuracy_report_phase3.md 생성

---

**문서 작성**: Claude Code Assistant
**최종 업데이트**: 2025-10-29
**버전**: 1.0
**상태**: ✅ Phase 2 완료, Phase 3 준비 완료
