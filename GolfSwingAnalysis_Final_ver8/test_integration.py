"""
골프 스윙 분석 시스템 통합 테스트
Author: Maxform 개발팀
Description: 전체 시스템 통합 테스트 및 성능 검증
"""

import pytest
import numpy as np
import cv2
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from golf_swing_analyzer import GolfSwingAnalyzer, SystemConfig, IRLightController, AnalysisResult
from stereo_vision import StereoVisionSystem
from object_tracker import ShotDetector, BallTracker, ClubTracker

class TestSystemConfig:
    """시스템 설정 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = SystemConfig()
        
        assert config.camera_fps == 240
        assert config.roi_ball == (200, 150, 400, 300)
        assert config.roi_club == (100, 100, 600, 400)
        assert config.shot_detection_threshold == 30
        assert config.min_tracking_points == 10
        assert config.max_tracking_time == 2.0
        assert config.calibration_file == "stereo_calibration.json"
    
    def test_custom_config(self):
        """사용자 정의 설정 테스트"""
        config = SystemConfig(
            camera_fps=120,
            roi_ball=(100, 100, 200, 200),
            shot_detection_threshold=50,
            min_tracking_points=20
        )
        
        assert config.camera_fps == 120
        assert config.roi_ball == (100, 100, 200, 200)
        assert config.shot_detection_threshold == 50
        assert config.min_tracking_points == 20

class TestIRLightController:
    """IR 조명 제어기 테스트"""
    
    def test_initialization(self):
        """초기화 테스트"""
        controller = IRLightController()
        assert controller.enable_ir
        assert not controller.is_on
        
        controller_disabled = IRLightController(enable_ir=False)
        assert not controller_disabled.enable_ir
    
    def test_turn_on_off(self):
        """조명 켜기/끄기 테스트"""
        controller = IRLightController()
        
        controller.turn_on()
        assert controller.is_on
        
        controller.turn_off()
        assert not controller.is_on
    
    def test_turn_on_off_disabled(self):
        """비활성화된 상태에서 조명 제어 테스트"""
        controller = IRLightController(enable_ir=False)
        
        controller.turn_on()
        assert not controller.is_on  # 비활성화되어 있으므로 켜지지 않음
        
        controller.turn_off()
        assert not controller.is_on
    
    def test_synchronized_capture(self):
        """동기화된 촬영 테스트"""
        controller = IRLightController()
        
        # 모의 촬영 함수
        capture_called = False
        def mock_capture():
            nonlocal capture_called
            capture_called = True
            return "captured_frame"
        
        result = controller.synchronized_capture(mock_capture)
        
        assert capture_called
        assert result == "captured_frame"
        assert not controller.is_on  # 촬영 후 꺼져야 함

class TestGolfSwingAnalyzer:
    """골프 스윙 분석기 통합 테스트"""
    
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        self.config = SystemConfig()
        self.analyzer = GolfSwingAnalyzer(self.config)
    
    def test_initialization(self):
        """초기화 테스트"""
        analyzer = GolfSwingAnalyzer()
        
        # 기본 설정 확인
        assert analyzer.config is not None
        assert isinstance(analyzer.stereo_system, StereoVisionSystem)
        assert isinstance(analyzer.shot_detector, ShotDetector)
        assert isinstance(analyzer.ball_tracker, BallTracker)
        assert isinstance(analyzer.club_tracker, ClubTracker)
        assert isinstance(analyzer.ir_controller, IRLightController)
        
        # 상태 변수 확인
        assert not analyzer.is_running
        assert not analyzer.is_analyzing
        assert analyzer.current_analysis is None
        assert analyzer.analysis_results == []
        assert analyzer.frame_count == 0
    
    def test_initialization_with_config(self):
        """설정과 함께 초기화 테스트"""
        custom_config = SystemConfig(camera_fps=120)
        analyzer = GolfSwingAnalyzer(custom_config)
        
        assert analyzer.config.camera_fps == 120
        assert analyzer.shot_detector.threshold == custom_config.shot_detection_threshold
    
    @patch('cv2.VideoCapture')
    def test_initialize_cameras_success(self, mock_video_capture):
        """카메라 초기화 성공 테스트"""
        # 모의 카메라 객체 설정
        mock_camera = MagicMock()
        mock_camera.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_camera
        
        result = self.analyzer.initialize_cameras(0, 1)
        
        assert result
        assert self.analyzer.left_camera is not None
        assert self.analyzer.right_camera is not None
        
        # 카메라 설정 호출 확인
        mock_camera.set.assert_called()
    
    @patch('cv2.VideoCapture')
    def test_initialize_cameras_failure(self, mock_video_capture):
        """카메라 초기화 실패 테스트"""
        # 모의 카메라 객체 설정 (읽기 실패)
        mock_camera = MagicMock()
        mock_camera.read.return_value = (False, None)
        mock_video_capture.return_value = mock_camera
        
        result = self.analyzer.initialize_cameras(0, 1)
        
        assert not result
    
    def test_calibrate_system_no_path(self):
        """존재하지 않는 경로로 캘리브레이션 테스트"""
        result = self.analyzer.calibrate_system("nonexistent_path")
        assert not result
    
    def test_capture_frame_pair_no_cameras(self):
        """카메라가 없는 상태에서 프레임 캡처 테스트"""
        left_frame, right_frame = self.analyzer.capture_frame_pair()
        
        assert left_frame is None
        assert right_frame is None
    
    def test_start_analysis(self):
        """분석 시작 테스트"""
        assert not self.analyzer.is_analyzing
        
        self.analyzer.start_analysis()
        
        assert self.analyzer.is_analyzing
        assert hasattr(self.analyzer, 'analysis_start_time')
        
        # 추적기들이 리셋되었는지 확인
        assert len(self.analyzer.ball_tracker.trajectory_points) == 0
        assert len(self.analyzer.club_tracker.trajectory_points) == 0
    
    def test_complete_analysis_not_analyzing(self):
        """분석 중이 아닐 때 완료 시도 테스트"""
        assert not self.analyzer.is_analyzing
        
        self.analyzer.complete_analysis()
        
        # 상태가 변경되지 않아야 함
        assert not self.analyzer.is_analyzing
        assert self.analyzer.current_analysis is None
    
    def test_complete_analysis_success(self):
        """분석 완료 성공 테스트"""
        # 분석 시작
        self.analyzer.start_analysis()
        
        # 일부 추적 데이터 추가
        for i in range(15):
            pos = np.array([i*0.1, i*0.05, i*0.02], dtype=float)
            timestamp = time.time() + i*0.004
            self.analyzer.ball_tracker.update_tracking(pos, timestamp)
            self.analyzer.club_tracker.update_tracking(pos, timestamp)
        
        self.analyzer.complete_analysis()
        
        assert not self.analyzer.is_analyzing
        assert self.analyzer.current_analysis is not None
        assert len(self.analyzer.analysis_results) == 1
        assert isinstance(self.analyzer.current_analysis, AnalysisResult)
    
    def test_process_frame_pair_basic(self):
        """기본 프레임 쌍 처리 테스트"""
        left_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = self.analyzer.process_frame_pair(left_frame, right_frame)
        
        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'shot_detected' in result
        assert 'ball_detected' in result
        assert 'club_detected' in result
        assert 'processing_time' in result
        assert result['processing_time'] > 0
    
    def test_process_frame_pair_with_shot_detection(self):
        """샷 감지가 있는 프레임 처리 테스트"""
        left_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 샷 감지기를 모의로 설정
        with patch.object(self.analyzer.shot_detector, 'detect_shot', return_value=True):
            result = self.analyzer.process_frame_pair(left_frame, right_frame)
            
            assert result['shot_detected']
            assert self.analyzer.is_analyzing
    
    def test_get_system_status(self):
        """시스템 상태 반환 테스트"""
        status = self.analyzer.get_system_status()
        
        assert isinstance(status, dict)
        assert 'is_running' in status
        assert 'is_analyzing' in status
        assert 'is_calibrated' in status
        assert 'frame_count' in status
        assert 'analysis_count' in status
        assert 'camera_info' in status
        
        assert not status['is_running']
        assert not status['is_analyzing']
        assert not status['is_calibrated']
        assert status['frame_count'] == 0
        assert status['analysis_count'] == 0
    
    def test_save_results(self):
        """결과 저장 테스트"""
        # 가짜 분석 결과 생성
        from object_tracker import BallData, ClubData
        
        ball_data = BallData(45.0, 12.5, 2.3, 2500.0, 300.0, 85.2)
        club_data = ClubData(35.0, -2.1, 1.5, 0.8, -0.7)
        
        analysis_result = AnalysisResult(
            timestamp="2025-01-23T10:00:00",
            ball_data=ball_data,
            club_data=club_data,
            shot_detected=True,
            processing_time=1.5,
            frame_count=100
        )
        
        self.analyzer.analysis_results.append(analysis_result)
        
        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.analyzer.save_results(temp_file)
            
            # 파일이 생성되었는지 확인
            assert os.path.exists(temp_file)
            
            # 파일 내용 확인
            import json
            with open(temp_file, 'r') as f:
                saved_data = json.load(f)
            
            assert len(saved_data) == 1
            assert saved_data[0]['timestamp'] == "2025-01-23T10:00:00"
            assert saved_data[0]['shot_detected'] is True
            assert saved_data[0]['ball_data']['speed'] == 45.0
            assert saved_data[0]['club_data']['speed'] == 35.0
            
        finally:
            # 임시 파일 정리
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_stop_analysis(self):
        """분석 중지 테스트"""
        # 상태 설정
        self.analyzer.is_running = True
        self.analyzer.is_analyzing = True
        
        # 모의 카메라 설정
        self.analyzer.left_camera = MagicMock()
        self.analyzer.right_camera = MagicMock()
        
        with patch('cv2.destroyAllWindows') as mock_destroy:
            self.analyzer.stop_analysis()
            
            assert not self.analyzer.is_running
            assert not self.analyzer.is_analyzing
            self.analyzer.left_camera.release.assert_called_once()
            self.analyzer.right_camera.release.assert_called_once()
            mock_destroy.assert_called_once()
    
    def test_callback_functions(self):
        """콜백 함수 테스트"""
        shot_detected_called = False
        analysis_complete_called = False
        frame_processed_called = False
        
        def on_shot_detected():
            nonlocal shot_detected_called
            shot_detected_called = True
        
        def on_analysis_complete(result):
            nonlocal analysis_complete_called
            analysis_complete_called = True
        
        def on_frame_processed(result):
            nonlocal frame_processed_called
            frame_processed_called = True
        
        # 콜백 함수 설정
        self.analyzer.on_shot_detected = on_shot_detected
        self.analyzer.on_analysis_complete = on_analysis_complete
        self.analyzer.on_frame_processed = on_frame_processed
        
        # 프레임 처리 (콜백 호출 확인)
        left_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        self.analyzer.process_frame_pair(left_frame, right_frame)
        
        assert frame_processed_called
        
        # 샷 감지 콜백 테스트
        with patch.object(self.analyzer.shot_detector, 'detect_shot', return_value=True):
            self.analyzer.process_frame_pair(left_frame, right_frame)
            assert shot_detected_called
    
    def test_visualize_frame(self):
        """프레임 시각화 테스트"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = {
            'shot_detected': True,
            'ball_detected': True,
            'club_detected': False,
            'processing_time': 0.05
        }
        
        # cv2.imshow를 모의로 설정
        with patch('cv2.imshow') as mock_imshow, \
             patch('cv2.putText') as mock_puttext, \
             patch('cv2.rectangle') as mock_rectangle:
            
            self.analyzer.visualize_frame(frame, result)
            
            # 시각화 함수들이 호출되었는지 확인
            mock_imshow.assert_called_once()
            mock_puttext.assert_called()
            mock_rectangle.assert_called()

class TestEndToEndIntegration:
    """종단간 통합 테스트"""
    
    def test_complete_workflow_simulation(self):
        """완전한 워크플로우 시뮬레이션 테스트"""
        config = SystemConfig(min_tracking_points=5, max_tracking_time=0.5)
        analyzer = GolfSwingAnalyzer(config)
        
        # 시뮬레이션된 프레임 시퀀스
        frames = []
        for i in range(20):
            left_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            right_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append((left_frame, right_frame))
        
        # 분석 시작 트리거
        analyzer.start_analysis()
        
        # 프레임들을 순차적으로 처리
        for i, (left_frame, right_frame) in enumerate(frames):
            result = analyzer.process_frame_pair(left_frame, right_frame)
            
            # 일부 프레임에서는 볼과 클럽이 감지된 것으로 시뮬레이션
            if 5 <= i <= 15:
                # 가짜 3D 위치 데이터 추가
                ball_pos = np.array([i*0.1, i*0.05, i*0.02], dtype=float)
                club_pos = np.array([i*0.08, i*0.03, -i*0.01], dtype=float)
                
                analyzer.ball_tracker.update_tracking(ball_pos, time.time() + i*0.004)
                analyzer.club_tracker.update_tracking(club_pos, time.time() + i*0.004)
            
            time.sleep(0.001)  # 실제 프레임 간격 시뮬레이션
        
        # 분석 완료
        analyzer.complete_analysis()
        
        # 결과 확인
        assert not analyzer.is_analyzing
        assert analyzer.current_analysis is not None
        assert len(analyzer.analysis_results) == 1
        
        # 분석 결과 데이터 확인
        result = analyzer.current_analysis
        assert result.ball_data is not None or result.club_data is not None
        assert result.shot_detected
        assert result.processing_time > 0

if __name__ == "__main__":
    # 통합 테스트 실행
    pytest.main([__file__, "-v", "--cov=golf_swing_analyzer", "--cov-report=html"])

