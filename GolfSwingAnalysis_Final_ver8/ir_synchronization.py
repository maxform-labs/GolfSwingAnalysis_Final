"""
IR 조명 동기화 시스템
Author: Maxform 개발팀
Description: 키오스크형 골프 스윙 분석을 위한 IR 조명과 카메라 동기화 시스템
- 240fps 동기화 지원
- 자동 샷 감지 트리거
- PWM 제어 기반 조명 강도 조절
"""

import cv2
import numpy as np
import time
import threading
import serial
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import queue

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IRTriggerMode(Enum):
    """IR 트리거 모드"""
    MANUAL = "manual"
    AUTO_SHOT_DETECTION = "auto_shot"
    CONTINUOUS = "continuous"
    FRAME_SYNC = "frame_sync"

@dataclass
class IRConfig:
    """IR 조명 설정"""
    wavelength: int = 850  # nm
    intensity: int = 100  # 0-100%
    pulse_duration_ms: int = 5  # 펄스 지속 시간
    trigger_mode: IRTriggerMode = IRTriggerMode.AUTO_SHOT_DETECTION
    serial_port: str = "/dev/ttyUSB0"  # Linux 기본값
    baud_rate: int = 115200
    sync_delay_us: int = 100  # 동기화 지연 (마이크로초)
    auto_exposure_compensation: bool = True  # 자동 노출 보정
    adaptive_intensity: bool = True  # 적응형 조명 강도

class IRLightController:
    """IR 조명 제어 클래스"""
    
    def __init__(self, config: IRConfig):
        """
        IR 조명 제어기 초기화
        
        Args:
            config: IR 설정
        """
        self.config = config
        self.serial_connection = None
        self.is_connected = False
        self.is_on = False
        self.current_intensity = 0
        
        # 성능 모니터링
        self.sync_timing_history = []
        
        # 적응형 제어
        self.ambient_light_level = 0
        self.target_brightness = 200  # 목표 골프공 밝기
        
        # 연결 시도
        self._connect_to_controller()
    
    def _connect_to_controller(self):
        """IR 조명 컨트롤러 연결"""
        try:
            self.serial_connection = serial.Serial(
                port=self.config.serial_port,
                baudrate=self.config.baud_rate,
                timeout=1.0
            )
            self.is_connected = True
            logger.info(f"IR 조명 컨트롤러 연결 성공: {self.config.serial_port}")
            
            # 초기 설정 전송
            self._send_initial_config()
            
        except serial.SerialException as e:
            logger.warning(f"IR 조명 컨트롤러 연결 실패: {e}")
            self.is_connected = False
    
    def _send_initial_config(self):
        """초기 설정 전송"""
        if not self.is_connected:
            return
        
        try:
            # 설정 명령 전송
            config_cmd = f"CONFIG:{self.config.wavelength},{self.config.pulse_duration_ms},{self.config.sync_delay_us}\n"
            self.serial_connection.write(config_cmd.encode())
            
            # 응답 확인
            response = self.serial_connection.readline().decode().strip()
            if response == "CONFIG_OK":
                logger.info("IR 조명 초기 설정 완료")
            else:
                logger.warning(f"IR 조명 설정 응답 오류: {response}")
                
        except Exception as e:
            logger.error(f"IR 조명 초기 설정 실패: {e}")
    
    def turn_on(self, intensity: Optional[int] = None):
        """
        IR 조명 켜기
        
        Args:
            intensity: 조명 강도 (0-100%), None이면 설정값 사용
        """
        if intensity is None:
            intensity = self.config.intensity
        
        intensity = max(0, min(100, intensity))  # 0-100% 범위 제한
        
        if self.is_connected:
            try:
                cmd = f"ON:{intensity}\n"
                self.serial_connection.write(cmd.encode())
                
                response = self.serial_connection.readline().decode().strip()
                if response == "ON_OK":
                    self.is_on = True
                    self.current_intensity = intensity
                    logger.info(f"IR 조명 켜짐 - 강도: {intensity}%")
                else:
                    logger.warning(f"IR 조명 켜기 실패: {response}")
                    
            except Exception as e:
                logger.error(f"IR 조명 제어 오류: {e}")
        else:
            # 시뮬레이션 모드
            self.is_on = True
            self.current_intensity = intensity
            logger.info(f"IR 조명 켜짐 (시뮬레이션) - 강도: {intensity}%")
    
    def turn_off(self):
        """IR 조명 끄기"""
        if self.is_connected:
            try:
                self.serial_connection.write(b"OFF\n")
                
                response = self.serial_connection.readline().decode().strip()
                if response == "OFF_OK":
                    self.is_on = False
                    self.current_intensity = 0
                    logger.info("IR 조명 꺼짐")
                else:
                    logger.warning(f"IR 조명 끄기 실패: {response}")
                    
            except Exception as e:
                logger.error(f"IR 조명 제어 오류: {e}")
        else:
            # 시뮬레이션 모드
            self.is_on = False
            self.current_intensity = 0
            logger.info("IR 조명 꺼짐 (시뮬레이션)")
    
    def pulse(self, intensity: Optional[int] = None, duration_ms: Optional[int] = None):
        """
        IR 조명 펄스
        
        Args:
            intensity: 펄스 강도
            duration_ms: 펄스 지속 시간
        """
        if intensity is None:
            intensity = self.config.intensity
        if duration_ms is None:
            duration_ms = self.config.pulse_duration_ms
        
        if self.is_connected:
            try:
                cmd = f"PULSE:{intensity},{duration_ms}\n"
                self.serial_connection.write(cmd.encode())
                
                response = self.serial_connection.readline().decode().strip()
                if response == "PULSE_OK":
                    logger.debug(f"IR 펄스 실행 - 강도: {intensity}%, 지속시간: {duration_ms}ms")
                else:
                    logger.warning(f"IR 펄스 실패: {response}")
                    
            except Exception as e:
                logger.error(f"IR 펄스 제어 오류: {e}")
        else:
            # 시뮬레이션 모드
            logger.debug(f"IR 펄스 실행 (시뮬레이션) - 강도: {intensity}%, 지속시간: {duration_ms}ms")
    
    def adjust_intensity_adaptive(self, frame: np.ndarray) -> int:
        """
        적응형 조명 강도 조절
        
        Args:
            frame: 현재 프레임
            
        Returns:
            조정된 강도
        """
        if not self.config.adaptive_intensity:
            return self.current_intensity
        
        # 주변 광량 측정
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.ambient_light_level = np.mean(gray)
        
        # 골프공 영역의 밝기 측정 (중앙 영역 가정)
        h, w = gray.shape
        center_roi = gray[h//3:2*h//3, w//3:2*w//3]
        ball_brightness = np.max(center_roi)
        
        # 목표 밝기와 비교하여 강도 조절
        brightness_ratio = self.target_brightness / max(ball_brightness, 1)
        
        if brightness_ratio > 1.2:  # 너무 어두움
            new_intensity = min(100, self.current_intensity + 10)
        elif brightness_ratio < 0.8:  # 너무 밝음
            new_intensity = max(20, self.current_intensity - 10)
        else:
            new_intensity = self.current_intensity
        
        if new_intensity != self.current_intensity:
            self.turn_on(new_intensity)
            logger.debug(f"적응형 조명 강도 조절: {self.current_intensity}% -> {new_intensity}%")
        
        return new_intensity
    
    def get_status(self) -> Dict[str, Any]:
        """IR 조명 상태 반환"""
        return {
            'is_connected': self.is_connected,
            'is_on': self.is_on,
            'current_intensity': self.current_intensity,
            'ambient_light_level': self.ambient_light_level,
            'config': self.config
        }
    
    def close(self):
        """연결 종료"""
        if self.is_connected and self.serial_connection:
            self.turn_off()
            self.serial_connection.close()
            self.is_connected = False
            logger.info("IR 조명 컨트롤러 연결 종료")

class IRSynchronizationSystem:
    """IR 조명 동기화 시스템"""
    
    def __init__(self, config: IRConfig):
        """
        IR 동기화 시스템 초기화
        
        Args:
            config: IR 설정
        """
        self.config = config
        self.ir_controller = IRLightController(config)
        
        # 동기화 상태
        self.is_syncing = False
        self.sync_thread = None
        self.frame_sync_queue = queue.Queue(maxsize=10)
        
        # 콜백 함수들
        self.shot_detection_callback = None
        self.frame_sync_callback = None
        
        # 성능 모니터링
        self.sync_performance = {
            'total_syncs': 0,
            'failed_syncs': 0,
            'average_delay_us': 0,
            'max_delay_us': 0
        }
    
    def start_synchronization(self):
        """동기화 시작"""
        if self.is_syncing:
            logger.warning("이미 동기화가 실행 중입니다")
            return
        
        self.is_syncing = True
        
        if self.config.trigger_mode == IRTriggerMode.FRAME_SYNC:
            self.sync_thread = threading.Thread(target=self._frame_sync_loop, daemon=True)
            self.sync_thread.start()
            logger.info("프레임 동기화 모드 시작")
        
        elif self.config.trigger_mode == IRTriggerMode.CONTINUOUS:
            self.ir_controller.turn_on()
            logger.info("연속 조명 모드 시작")
        
        elif self.config.trigger_mode == IRTriggerMode.AUTO_SHOT_DETECTION:
            logger.info("자동 샷 감지 모드 시작")
        
        else:  # MANUAL
            logger.info("수동 제어 모드 시작")
    
    def stop_synchronization(self):
        """동기화 중지"""
        self.is_syncing = False
        
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=1.0)
        
        self.ir_controller.turn_off()
        logger.info("IR 동기화 중지")
    
    def _frame_sync_loop(self):
        """프레임 동기화 루프"""
        target_interval = 1.0 / 240.0  # 240fps 간격
        
        while self.is_syncing:
            start_time = time.perf_counter()
            
            # IR 펄스 트리거
            self.ir_controller.pulse()
            
            # 프레임 동기화 콜백 호출
            if self.frame_sync_callback:
                self.frame_sync_callback()
            
            # 성능 모니터링
            elapsed = time.perf_counter() - start_time
            self._update_sync_performance(elapsed * 1000000)  # 마이크로초 변환
            
            # 다음 프레임까지 대기
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def trigger_shot_detection(self, frame: np.ndarray):
        """
        샷 감지 트리거
        
        Args:
            frame: 현재 프레임
        """
        if self.config.trigger_mode != IRTriggerMode.AUTO_SHOT_DETECTION:
            return
        
        start_time = time.perf_counter()
        
        # 적응형 조명 강도 조절
        if self.config.adaptive_intensity:
            self.ir_controller.adjust_intensity_adaptive(frame)
        
        # IR 펄스 트리거
        self.ir_controller.pulse()
        
        # 샷 감지 콜백 호출
        if self.shot_detection_callback:
            self.shot_detection_callback(frame)
        
        # 성능 모니터링
        elapsed = time.perf_counter() - start_time
        self._update_sync_performance(elapsed * 1000000)
    
    def manual_trigger(self, intensity: Optional[int] = None, duration_ms: Optional[int] = None):
        """
        수동 트리거
        
        Args:
            intensity: 조명 강도
            duration_ms: 지속 시간
        """
        if self.config.trigger_mode != IRTriggerMode.MANUAL:
            logger.warning("수동 모드가 아닙니다")
            return
        
        self.ir_controller.pulse(intensity, duration_ms)
    
    def set_shot_detection_callback(self, callback: Callable[[np.ndarray], None]):
        """샷 감지 콜백 설정"""
        self.shot_detection_callback = callback
    
    def set_frame_sync_callback(self, callback: Callable[[], None]):
        """프레임 동기화 콜백 설정"""
        self.frame_sync_callback = callback
    
    def _update_sync_performance(self, delay_us: float):
        """동기화 성능 업데이트"""
        self.sync_performance['total_syncs'] += 1
        
        if delay_us > 5000:  # 5ms 이상이면 실패로 간주
            self.sync_performance['failed_syncs'] += 1
        
        # 평균 지연 시간 계산
        total = self.sync_performance['total_syncs']
        current_avg = self.sync_performance['average_delay_us']
        self.sync_performance['average_delay_us'] = (current_avg * (total - 1) + delay_us) / total
        
        # 최대 지연 시간 업데이트
        if delay_us > self.sync_performance['max_delay_us']:
            self.sync_performance['max_delay_us'] = delay_us
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.sync_performance.copy()
        
        if stats['total_syncs'] > 0:
            stats['success_rate'] = (stats['total_syncs'] - stats['failed_syncs']) / stats['total_syncs'] * 100
        else:
            stats['success_rate'] = 0
        
        stats['ir_controller_status'] = self.ir_controller.get_status()
        
        return stats
    
    def calibrate_sync_delay(self, test_frames: int = 100) -> float:
        """
        동기화 지연 캘리브레이션
        
        Args:
            test_frames: 테스트 프레임 수
            
        Returns:
            최적 지연 시간 (마이크로초)
        """
        logger.info(f"동기화 지연 캘리브레이션 시작 ({test_frames} 프레임)")
        
        delays = []
        
        for i in range(test_frames):
            start_time = time.perf_counter()
            
            # IR 펄스 트리거
            self.ir_controller.pulse()
            
            # 지연 시간 측정
            end_time = time.perf_counter()
            delay_us = (end_time - start_time) * 1000000
            delays.append(delay_us)
            
            # 240fps 간격으로 대기
            time.sleep(1.0 / 240.0)
        
        # 통계 계산
        avg_delay = np.mean(delays)
        std_delay = np.std(delays)
        optimal_delay = avg_delay + std_delay  # 안전 마진 추가
        
        logger.info(f"캘리브레이션 완료 - 평균: {avg_delay:.1f}μs, 표준편차: {std_delay:.1f}μs")
        logger.info(f"최적 지연 시간: {optimal_delay:.1f}μs")
        
        # 설정 업데이트
        self.config.sync_delay_us = int(optimal_delay)
        
        return optimal_delay
    
    def close(self):
        """시스템 종료"""
        self.stop_synchronization()
        self.ir_controller.close()
        logger.info("IR 동기화 시스템 종료")

# 사용 예제
if __name__ == "__main__":
    # IR 설정
    config = IRConfig(
        wavelength=850,
        intensity=80,
        trigger_mode=IRTriggerMode.AUTO_SHOT_DETECTION,
        adaptive_intensity=True
    )
    
    # 동기화 시스템 초기화
    sync_system = IRSynchronizationSystem(config)
    
    try:
        # 동기화 시작
        sync_system.start_synchronization()
        
        # 캘리브레이션 수행
        optimal_delay = sync_system.calibrate_sync_delay(50)
        
        print(f"최적 동기화 지연: {optimal_delay:.1f}μs")
        
        # 성능 통계 출력
        stats = sync_system.get_performance_stats()
        print(f"동기화 성공률: {stats['success_rate']:.1f}%")
        print(f"평균 지연: {stats['average_delay_us']:.1f}μs")
        
        # 테스트 실행
        print("IR 동기화 테스트 실행 중... (Ctrl+C로 종료)")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n테스트 종료")
    finally:
        sync_system.close()

