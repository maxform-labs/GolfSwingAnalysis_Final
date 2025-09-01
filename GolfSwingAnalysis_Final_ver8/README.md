##골프 스윙 분석 시스템 - 코드베이스 및 워크플로우
# 파일 구조

```
GolfSwingAnalysis_Final_ver1_2025-08-03/
├── README_Final_ver1.md                    # 이 파일
├── 개발이력관리.md                          # 개발 이력 관리
├── stereo_vision_vertical.py               # 수직 스테레오 비전 시스템
├── golf_swing_analyzer.py                  # 골프 스윙 분석기
├── object_tracker.py                       # 객체 추적 시스템
├── ir_synchronization.py                   # IR 동기화 시스템
├── kiosk_system.py                         # 키오스크 시스템
├── web_dashboard.py                        # 웹 대시보드
├── simulation_environment.py               # 시뮬레이션 환경
├── test_integration.py                     # 통합 테스트
├── advanced_algorithms_ver4.py             # 고급 알고리즘 모듈
├── realistic_achievement_system.py                  # 95% 정확도 검증 시스템
├── realistic_3d_validation_app.py          # 3D 검증 앱
└── requirements.txt                        # 필요 패키지 목록
```

##시스템 아키텍처 및 실행 흐름

  #실행 진입점 (Entry Points)

  프로젝트는 여러 실행 모드를 제공합니다:

  # 1. 메인 분석 시스템
  python golf_swing_analyzer.py

  # 2. 키오스크 GUI 시스템
  python kiosk_system.py

  # 3. 웹 대시보드
  python web_dashboard.py

  # 4. 시뮬레이션 환경 (테스트용)
  python simulation_environment.py

  # 핵심 워크플로우

  [카메라 입력] → [프레임 캡처] → [객체 검출] → [3D 분석] → [결과 출력]
       ↓              ↓              ↓              ↓           ↓
    240fps       멀티스레딩      IR 동기화    스테레오비전   데이터저장

  #멀티스레딩 파이프라인 구조

  RealTimeProcessor (golf_swing_analyzer.py)가 4개의 병렬 스레드로 작동:

  1. frame_capture_thread (1ms)
    - 상/하 카메라에서 동기화된 프레임 캡처
    - 240fps 유지
  2. object_detection_thread (5ms)
    - 골프공/클럽 검출
    - ROI 기반 추적
  3. stereo_analysis_thread (8ms)
    - Y축 시차 계산
    - 3D 좌표 변환
    - 칼만 필터 적용
  4. result_output_thread (2ms)
    - 데이터 검증
    - DB 저장
    - UI 업데이트

  #모듈 간 데이터 흐름

  # 1. 시스템 초기화
  config = SystemConfig(camera_fps=240)
  analyzer = GolfSwingAnalyzer(config)

  # 2. 스테레오 비전 설정
  stereo_config = VerticalStereoConfig(
      vertical_baseline=400.0,  # mm
      inward_angle=12.0        # degrees
  )
  stereo_vision = VerticalStereoVision(stereo_config)

  # 3. IR 조명 동기화
  ir_system = IRSynchronizationSystem(
      IRConfig(intensity=80, trigger_mode="auto_shot")
  )

  # 4. 객체 추적기
  ball_tracker = BallTracker()
  club_tracker = ClubTracker()

  # 5. 메인 루프
  analyzer.start_analysis()  # 분석 시작

  #키오스크 시스템 워크플로우

  KioskGUI (kiosk_system.py)는 터치스크린 인터페이스 제공:

  # GUI 초기화
  kiosk = KioskGUI()

  # 이벤트 루프
  ├── 사용자 등록/로그인
  ├── 캘리브레이션 모드
  ├── 실시간 분석 시작
  │   ├── 카메라 피드 표시
  │   ├── 3D 시각화
  │   └── 실시간 데이터 표시
  ├── 결과 저장 (SQLite DB)
  └── 리포트 생성

  #웹 대시보드 워크플로우

  WebDashboard (web_dashboard.py)는 Flask 서버로 실행:

  # 서버 시작
  dashboard = WebDashboard()
  dashboard.run(host='0.0.0.0', port=5000)

  # API 엔드포인트
  ├── /api/status     - 시스템 상태
  ├── /api/start      - 분석 시작
  ├── /api/stop       - 분석 중지
  ├── /api/results    - 결과 조회
  └── /api/calibrate  - 캘리브레이션

  #데이터 처리 플로우

  1. 프레임 캡처 (카메라)
     ↓
  2. 전처리 (노이즈 제거, IR 백그라운드 차감)
     ↓
  3. 객체 검출 (공/클럽 위치)
     ↓
  4. 스테레오 매칭 (Y축 시차)
     ↓
  5. 3D 좌표 계산
     ↓
  6. 칼만 필터링 (노이즈 감소)
     ↓
  7. 물리 계산 (속도, 각도, 스핀)
     ↓
  8. 검증 (물리적 일관성)
     ↓
  9. 결과 출력 (UI, DB, API)

  #성능 최적화 전략

  - 메모리 풀: 프레임 버퍼 재사용
  - GPU 가속: CUDA 지원 (OpenCV)
  - 적응형 ROI: 동적 관심영역
  - 큐 관리: 비동기 처리

  # 시스템 시작/종료 프로세스

  시작:
  1. 설정 파일 로드 (config.json)
  2. 카메라 초기화 및 검증
  3. IR 조명 시스템 활성화
  4. 캘리브레이션 데이터 로드
  5. 멀티스레딩 파이프라인 시작
  6. UI/웹서버 시작

  종료:
  1. 처리 스레드 종료 신호
  2. 큐 비우기 및 대기
  3. 데이터 저장 완료
  4. 카메라/IR 시스템 해제
  5. 리소스 정리

  # 주요 특징

  - 실시간 처리: 240fps, 16ms 이하 레이턴시
  - 높은 정확도: ±5% 이내 측정 오차
  - 수직 스테레오: 공간 효율적 설계
  - 자동 캘리브레이션: 정확도 유지
  - 다중 인터페이스: GUI, 웹, API 지원

##설치 및 실행

# 1. 환경 설정
```bash
# 필요 패키지 설치
pip install -r requirements.txt

# OpenCV, NumPy, SciPy 등 주요 패키지 확인
python -c "import cv2, numpy, scipy; print('패키지 설치 완료')"
```

# 2. 시스템 실행
```bash
# 메인 골프 스윙 분석 시스템 실행
python golf_swing_analyzer.py

# 웹 대시보드 실행 (별도 터미널)
python web_dashboard.py

# 키오스크 시스템 실행
python kiosk_system.py
```

# 3. 검증 시스템 실행
```bash
# 95% 정확도 검증 시스템 실행
python realistic_achievement_system.py

# 3D 검증 앱 실행
python realistic_3d_validation_app.py
# 브라우저에서 http://localhost:5002 접속
```

# 고급 알고리즘 융합
- **고정밀 칼만 필터**: 프로세스/측정 노이즈 최적화
- **스마트 베이지안 추정**: 3개 추정기 앙상블
- **적응형 보정**: 스킬 레벨별 맞춤 보정
- **물리적 제약**: 물리 법칙 기반 타당성 검증