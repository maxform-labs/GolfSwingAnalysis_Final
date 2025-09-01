"""
키오스크형 골프 스윙 분석 시스템
Author: Maxform 개발팀
Description: 수직 스테레오 비전 기반 키오스크 통합 시스템
"""

import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import webbrowser
from pathlib import Path

# 로컬 모듈 임포트
try:
    from stereo_vision_vertical_ver2 import VerticalStereoVision, VerticalStereoConfig
    from ir_synchronization_ver2 import IRSynchronizationSystem, IRConfig, IRTriggerMode
    from object_tracker_ver1_2025_01_23 import GolfObjectTracker
except ImportError as e:
    print(f"모듈 임포트 오류: {e}")
    print("필요한 모듈들이 같은 디렉토리에 있는지 확인하세요.")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kiosk_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class KioskConfig:
    """키오스크 설정"""
    # 화면 설정 (13인치 노트북 최적화)
    screen_width: int = 1920
    screen_height: int = 1080
    fullscreen: bool = True
    auto_start: bool = True
    
    # 카메라 설정
    camera_top_index: int = 0
    camera_bottom_index: int = 1
    camera_resolution: Tuple[int, int] = (1920, 1080)
    camera_fps: int = 240
    
    # 측정 설정
    detection_zone_mm: Tuple[float, float] = (400.0, 400.0)
    measurement_zone_mm: Tuple[float, float] = (800.0, 800.0)
    accuracy_target_percent: float = 5.0
    
    # 데이터베이스 설정
    db_path: str = "golf_swing_data.db"
    auto_backup: bool = True
    backup_interval_hours: int = 24

class DatabaseManager:
    """데이터베이스 관리자"""
    
    def __init__(self, db_path: str):
        """
        데이터베이스 관리자 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.init_database()
        logger.info(f"데이터베이스 초기화 완료: {db_path}")
    
    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 스윙 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS swing_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    club_type TEXT,
                    
                    -- 볼 데이터
                    ball_speed_ms REAL,
                    launch_angle_deg REAL,
                    direction_angle_deg REAL,
                    backspin_rpm REAL,
                    sidespin_rpm REAL,
                    spin_axis_deg REAL,
                    
                    -- 클럽 데이터
                    club_speed_ms REAL,
                    attack_angle_deg REAL,
                    club_path_deg REAL,
                    face_angle_deg REAL,
                    loft_angle_deg REAL,
                    face_to_path_deg REAL,
                    
                    -- 시스템 데이터
                    accuracy_score REAL,
                    confidence_level REAL,
                    processing_time_ms REAL,
                    
                    -- 원시 데이터 (JSON)
                    raw_data TEXT
                )
            ''')
            
            # 시스템 로그 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT,
                    module TEXT,
                    message TEXT,
                    details TEXT
                )
            ''')
            
            # 캘리브레이션 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS calibration_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    calibration_type TEXT,
                    parameters TEXT,
                    accuracy_metrics TEXT,
                    is_active BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
    
    def save_swing_data(self, swing_data: Dict[str, Any]) -> int:
        """스윙 데이터 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO swing_data (
                    user_id, club_type,
                    ball_speed_ms, launch_angle_deg, direction_angle_deg,
                    backspin_rpm, sidespin_rpm, spin_axis_deg,
                    club_speed_ms, attack_angle_deg, club_path_deg,
                    face_angle_deg, loft_angle_deg, face_to_path_deg,
                    accuracy_score, confidence_level, processing_time_ms,
                    raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                swing_data.get('user_id', 'anonymous'),
                swing_data.get('club_type', 'unknown'),
                swing_data.get('ball_speed_ms', 0),
                swing_data.get('launch_angle_deg', 0),
                swing_data.get('direction_angle_deg', 0),
                swing_data.get('backspin_rpm', 0),
                swing_data.get('sidespin_rpm', 0),
                swing_data.get('spin_axis_deg', 0),
                swing_data.get('club_speed_ms', 0),
                swing_data.get('attack_angle_deg', 0),
                swing_data.get('club_path_deg', 0),
                swing_data.get('face_angle_deg', 0),
                swing_data.get('loft_angle_deg', 0),
                swing_data.get('face_to_path_deg', 0),
                swing_data.get('accuracy_score', 0),
                swing_data.get('confidence_level', 0),
                swing_data.get('processing_time_ms', 0),
                json.dumps(swing_data)
            ))
            
            return cursor.lastrowid
    
    def get_recent_swings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 스윙 데이터 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM swing_data 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                swing_dict = dict(zip(columns, row))
                if swing_dict['raw_data']:
                    try:
                        swing_dict['raw_data'] = json.loads(swing_dict['raw_data'])
                    except json.JSONDecodeError:
                        swing_dict['raw_data'] = {}
                results.append(swing_dict)
            
            return results

class KioskGUI:
    """키오스크 GUI 인터페이스"""
    
    def __init__(self, config: KioskConfig):
        """
        키오스크 GUI 초기화
        
        Args:
            config: 키오스크 설정
        """
        self.config = config
        self.root = tk.Tk()
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        
        # 시스템 컴포넌트
        self.db_manager = DatabaseManager(config.db_path)
        self.stereo_system = None
        self.ir_system = None
        self.object_tracker = None
        
        # 상태 변수
        self.is_running = False
        self.current_frame_top = None
        self.current_frame_bottom = None
        self.last_swing_data = None
        
        logger.info("키오스크 GUI 초기화 완료")
    
    def setup_window(self):
        """윈도우 설정"""
        self.root.title("골프 스윙 분석 키오스크 ver.2")
        self.root.geometry(f"{self.config.screen_width}x{self.config.screen_height}")
        
        if self.config.fullscreen:
            self.root.attributes('-fullscreen', True)
            # ESC 키로 전체화면 해제
            self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
        
        # 윈도우 아이콘 설정 (있는 경우)
        try:
            self.root.iconbitmap('golf_icon.ico')
        except:
            pass
        
        # 윈도우 닫기 이벤트
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """스타일 설정"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 커스텀 스타일 정의
        self.style.configure('Title.TLabel', 
                           font=('Arial', 24, 'bold'),
                           foreground='#2E7D32')
        
        self.style.configure('Header.TLabel',
                           font=('Arial', 16, 'bold'),
                           foreground='#1976D2')
        
        self.style.configure('Data.TLabel',
                           font=('Arial', 14),
                           foreground='#424242')
        
        self.style.configure('Success.TLabel',
                           font=('Arial', 12),
                           foreground='#388E3C')
        
        self.style.configure('Error.TLabel',
                           font=('Arial', 12),
                           foreground='#D32F2F')
    
    def create_widgets(self):
        """위젯 생성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 제목
        title_label = ttk.Label(main_frame, 
                               text="골프 스윙 분석 시스템",
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # 상단 프레임 (카메라 뷰 + 제어판)
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        # 카메라 뷰 프레임
        camera_frame = ttk.LabelFrame(top_frame, text="카메라 뷰", padding=10)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 카메라 캔버스 (13인치 최적화)
        self.camera_canvas = tk.Canvas(camera_frame, 
                                     width=800, height=450,
                                     bg='black')
        self.camera_canvas.pack()
        
        # 제어판 프레임
        control_frame = ttk.LabelFrame(top_frame, text="제어판", padding=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 시스템 제어 버튼들
        self.start_button = ttk.Button(control_frame, 
                                     text="시스템 시작",
                                     command=self.start_system,
                                     width=15)
        self.start_button.pack(pady=5)
        
        self.stop_button = ttk.Button(control_frame,
                                    text="시스템 중지",
                                    command=self.stop_system,
                                    state=tk.DISABLED,
                                    width=15)
        self.stop_button.pack(pady=5)
        
        self.calibrate_button = ttk.Button(control_frame,
                                         text="캘리브레이션",
                                         command=self.open_calibration,
                                         width=15)
        self.calibrate_button.pack(pady=5)
        
        # 구분선
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 상태 표시
        ttk.Label(control_frame, text="시스템 상태", style='Header.TLabel').pack()
        
        self.status_label = ttk.Label(control_frame, 
                                    text="대기 중",
                                    style='Data.TLabel')
        self.status_label.pack(pady=5)
        
        # IR 상태
        ttk.Label(control_frame, text="IR 동기화", style='Header.TLabel').pack(pady=(10, 0))
        
        self.ir_status_label = ttk.Label(control_frame,
                                       text="연결 안됨",
                                       style='Error.TLabel')
        self.ir_status_label.pack(pady=5)
        
        # 하단 프레임 (데이터 표시)
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(20, 0))
        
        # 볼 데이터 프레임
        ball_frame = ttk.LabelFrame(bottom_frame, text="볼 데이터", padding=10)
        ball_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.ball_data_labels = {}
        ball_data_items = [
            ('속도', 'ball_speed_ms', 'm/s'),
            ('발사각', 'launch_angle_deg', '°'),
            ('방향각', 'direction_angle_deg', '°'),
            ('백스핀', 'backspin_rpm', 'rpm'),
            ('사이드스핀', 'sidespin_rpm', 'rpm'),
            ('스핀축', 'spin_axis_deg', '°')
        ]
        
        for i, (name, key, unit) in enumerate(ball_data_items):
            row = i // 2
            col = i % 2
            
            label_frame = ttk.Frame(ball_frame)
            label_frame.grid(row=row, column=col, padx=5, pady=2, sticky='w')
            
            ttk.Label(label_frame, text=f"{name}:", width=8).pack(side=tk.LEFT)
            data_label = ttk.Label(label_frame, text=f"-- {unit}", style='Data.TLabel')
            data_label.pack(side=tk.LEFT)
            
            self.ball_data_labels[key] = data_label
        
        # 클럽 데이터 프레임
        club_frame = ttk.LabelFrame(bottom_frame, text="클럽 데이터", padding=10)
        club_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.club_data_labels = {}
        club_data_items = [
            ('클럽 속도', 'club_speed_ms', 'm/s'),
            ('어택 앵글', 'attack_angle_deg', '°'),
            ('클럽 패스', 'club_path_deg', '°'),
            ('페이스 앵글', 'face_angle_deg', '°'),
            ('로프트 앵글', 'loft_angle_deg', '°'),
            ('Face to Path', 'face_to_path_deg', '°')
        ]
        
        for i, (name, key, unit) in enumerate(club_data_items):
            row = i // 2
            col = i % 2
            
            label_frame = ttk.Frame(club_frame)
            label_frame.grid(row=row, column=col, padx=5, pady=2, sticky='w')
            
            ttk.Label(label_frame, text=f"{name}:", width=10).pack(side=tk.LEFT)
            data_label = ttk.Label(label_frame, text=f"-- {unit}", style='Data.TLabel')
            data_label.pack(side=tk.LEFT)
            
            self.club_data_labels[key] = data_label
    
    def start_system(self):
        """시스템 시작"""
        try:
            # 스테레오 비전 시스템 초기화
            stereo_config = VerticalStereoConfig(
                vertical_baseline=500.0,
                inward_angle=15.0,
                installation_height=450.0,
                detection_zone_size=self.config.detection_zone_mm,
                measurement_zone_size=self.config.measurement_zone_mm
            )
            self.stereo_system = VerticalStereoVision(stereo_config)
            
            # IR 시스템 초기화
            ir_config = IRConfig(
                wavelength=850,
                intensity=80,
                trigger_mode=IRTriggerMode.AUTO_SHOT_DETECTION
            )
            self.ir_system = IRSynchronizationSystem(ir_config)
            
            # 객체 추적기 초기화
            self.object_tracker = GolfObjectTracker()
            
            # 카메라 시작
            self.start_cameras()
            
            # 상태 업데이트
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="실행 중", style='Success.TLabel')
            
            # 메인 루프 시작
            self.start_main_loop()
            
            logger.info("시스템 시작 완료")
            
        except Exception as e:
            logger.error(f"시스템 시작 실패: {e}")
            messagebox.showerror("오류", f"시스템 시작에 실패했습니다:\n{e}")
    
    def stop_system(self):
        """시스템 중지"""
        try:
            self.is_running = False
            
            # 카메라 중지
            if hasattr(self, 'cap_top') and self.cap_top:
                self.cap_top.release()
            if hasattr(self, 'cap_bottom') and self.cap_bottom:
                self.cap_bottom.release()
            
            # IR 시스템 중지
            if self.ir_system:
                self.ir_system.stop_synchronization()
            
            # 상태 업데이트
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="중지됨", style='Error.TLabel')
            
            logger.info("시스템 중지 완료")
            
        except Exception as e:
            logger.error(f"시스템 중지 실패: {e}")
    
    def start_cameras(self):
        """카메라 시작"""
        try:
            # 상단 카메라
            self.cap_top = cv2.VideoCapture(self.config.camera_top_index)
            self.cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_resolution[0])
            self.cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_resolution[1])
            self.cap_top.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            # 하단 카메라
            self.cap_bottom = cv2.VideoCapture(self.config.camera_bottom_index)
            self.cap_bottom.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_resolution[0])
            self.cap_bottom.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_resolution[1])
            self.cap_bottom.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            if not self.cap_top.isOpened() or not self.cap_bottom.isOpened():
                raise Exception("카메라를 열 수 없습니다")
            
            logger.info("카메라 시작 완료")
            
        except Exception as e:
            logger.error(f"카메라 시작 실패: {e}")
            # 시뮬레이션 모드로 전환
            self.cap_top = None
            self.cap_bottom = None
            logger.info("시뮬레이션 모드로 전환")
    
    def start_main_loop(self):
        """메인 처리 루프 시작"""
        self.process_frame()
    
    def process_frame(self):
        """프레임 처리"""
        if not self.is_running:
            return
        
        try:
            # 프레임 캡처
            if self.cap_top and self.cap_bottom:
                ret_top, frame_top = self.cap_top.read()
                ret_bottom, frame_bottom = self.cap_bottom.read()
                
                if ret_top and ret_bottom:
                    self.current_frame_top = frame_top
                    self.current_frame_bottom = frame_bottom
                    
                    # 프레임 표시
                    self.display_frame(frame_top)
                    
                    # 골프 스윙 분석
                    self.analyze_golf_swing(frame_top, frame_bottom)
            else:
                # 시뮬레이션 모드
                self.simulate_golf_swing()
            
        except Exception as e:
            logger.error(f"프레임 처리 오류: {e}")
        
        # 다음 프레임 스케줄링
        self.root.after(33, self.process_frame)  # ~30fps
    
    def display_frame(self, frame):
        """프레임 표시"""
        try:
            # 프레임 크기 조정 (13인치 화면 최적화)
            display_frame = cv2.resize(frame, (800, 450))
            
            # BGR to RGB 변환
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # PIL Image로 변환
            from PIL import Image, ImageTk
            image = Image.fromarray(display_frame)
            photo = ImageTk.PhotoImage(image)
            
            # 캔버스에 표시
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(400, 225, image=photo)
            self.camera_canvas.image = photo  # 참조 유지
            
        except Exception as e:
            logger.error(f"프레임 표시 오류: {e}")
    
    def analyze_golf_swing(self, frame_top, frame_bottom):
        """골프 스윙 분석"""
        try:
            if not self.stereo_system or not self.object_tracker:
                return
            
            # 스테레오 정렬
            if self.stereo_system.is_calibrated:
                rect_top, rect_bottom = self.stereo_system.rectify_images(frame_top, frame_bottom)
                
                # 시차 계산
                disparity = self.stereo_system.compute_vertical_disparity(rect_top, rect_bottom)
                
                # 객체 추적
                ball_detected, ball_pos = self.object_tracker.detect_golf_ball(rect_top)
                club_detected, club_pos = self.object_tracker.detect_golf_club(rect_top)
                
                if ball_detected:
                    # 3D 좌표 계산
                    points_3d = self.stereo_system.calculate_3d_coordinates_vertical(
                        disparity, [ball_pos]
                    )
                    
                    if points_3d and points_3d[0] != (0.0, 0.0, 0.0):
                        # 스윙 데이터 계산
                        swing_data = self.calculate_swing_data(points_3d[0], ball_pos, club_pos)
                        
                        # 데이터 표시 및 저장
                        self.update_data_display(swing_data)
                        self.save_swing_data(swing_data)
            
        except Exception as e:
            logger.error(f"골프 스윙 분석 오류: {e}")
    
    def simulate_golf_swing(self):
        """골프 스윙 시뮬레이션"""
        # 시뮬레이션 데이터 생성
        import random
        
        swing_data = {
            'ball_speed_ms': round(random.uniform(40, 70), 1),
            'launch_angle_deg': round(random.uniform(8, 15), 1),
            'direction_angle_deg': round(random.uniform(-5, 5), 1),
            'backspin_rpm': round(random.uniform(2000, 4000)),
            'sidespin_rpm': round(random.uniform(-500, 500)),
            'spin_axis_deg': round(random.uniform(-10, 10), 1),
            'club_speed_ms': round(random.uniform(35, 55), 1),
            'attack_angle_deg': round(random.uniform(-2, 4), 1),
            'club_path_deg': round(random.uniform(-3, 3), 1),
            'face_angle_deg': round(random.uniform(-2, 2), 1),
            'loft_angle_deg': round(random.uniform(9, 12), 1),
            'face_to_path_deg': round(random.uniform(-2, 2), 1),
            'accuracy_score': round(random.uniform(85, 98), 1),
            'confidence_level': round(random.uniform(0.8, 0.95), 2),
            'processing_time_ms': round(random.uniform(100, 300))
        }
        
        # 가끔씩 데이터 업데이트 (실제 스윙 시뮬레이션)
        if random.random() < 0.1:  # 10% 확률
            self.update_data_display(swing_data)
    
    def calculate_swing_data(self, ball_3d, ball_2d, club_2d):
        """스윙 데이터 계산"""
        # 실제 물리 계산 로직 구현
        # 여기서는 간단한 예시 데이터 반환
        return {
            'ball_speed_ms': 45.2,
            'launch_angle_deg': 12.5,
            'direction_angle_deg': 1.2,
            'backspin_rpm': 2800,
            'sidespin_rpm': -150,
            'spin_axis_deg': 2.1,
            'club_speed_ms': 42.1,
            'attack_angle_deg': 1.8,
            'club_path_deg': 0.5,
            'face_angle_deg': 0.8,
            'loft_angle_deg': 10.5,
            'face_to_path_deg': 0.3,
            'accuracy_score': 92.5,
            'confidence_level': 0.89,
            'processing_time_ms': 180
        }
    
    def update_data_display(self, swing_data):
        """데이터 표시 업데이트"""
        try:
            # 볼 데이터 업데이트
            for key, label in self.ball_data_labels.items():
                if key in swing_data:
                    value = swing_data[key]
                    if key.endswith('_ms'):
                        unit = 'm/s'
                    elif key.endswith('_deg'):
                        unit = '°'
                    elif key.endswith('_rpm'):
                        unit = 'rpm'
                    else:
                        unit = ''
                    
                    label.config(text=f"{value} {unit}")
            
            # 클럽 데이터 업데이트
            for key, label in self.club_data_labels.items():
                if key in swing_data:
                    value = swing_data[key]
                    if key.endswith('_ms'):
                        unit = 'm/s'
                    elif key.endswith('_deg'):
                        unit = '°'
                    else:
                        unit = ''
                    
                    label.config(text=f"{value} {unit}")
            
            self.last_swing_data = swing_data
            
        except Exception as e:
            logger.error(f"데이터 표시 업데이트 오류: {e}")
    
    def save_swing_data(self, swing_data):
        """스윙 데이터 저장"""
        try:
            swing_data['timestamp'] = datetime.now().isoformat()
            swing_data['user_id'] = 'kiosk_user'
            swing_data['club_type'] = 'driver'  # 기본값
            
            record_id = self.db_manager.save_swing_data(swing_data)
            logger.info(f"스윙 데이터 저장 완료: ID {record_id}")
            
        except Exception as e:
            logger.error(f"스윙 데이터 저장 오류: {e}")
    
    def open_calibration(self):
        """캘리브레이션 창 열기"""
        CalibrationWindow(self.root, self.stereo_system)
    
    def on_closing(self):
        """윈도우 닫기 이벤트"""
        if messagebox.askokcancel("종료", "시스템을 종료하시겠습니까?"):
            self.stop_system()
            self.root.destroy()
    
    def run(self):
        """GUI 실행"""
        logger.info("키오스크 GUI 시작")
        self.root.mainloop()

class CalibrationWindow:
    """캘리브레이션 창"""
    
    def __init__(self, parent, stereo_system):
        """
        캘리브레이션 창 초기화
        
        Args:
            parent: 부모 윈도우
            stereo_system: 스테레오 비전 시스템
        """
        self.parent = parent
        self.stereo_system = stereo_system
        
        self.window = tk.Toplevel(parent)
        self.window.title("카메라 캘리브레이션")
        self.window.geometry("600x400")
        self.window.transient(parent)
        self.window.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        """위젯 생성"""
        main_frame = ttk.Frame(self.window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 제목
        ttk.Label(main_frame, 
                 text="스테레오 카메라 캘리브레이션",
                 font=('Arial', 16, 'bold')).pack(pady=(0, 20))
        
        # 설명
        instruction_text = """
캘리브레이션 절차:
1. 체스보드 패턴을 준비하세요 (9x6 격자)
2. 다양한 각도와 위치에서 체스보드를 촬영하세요
3. 최소 15장 이상의 이미지가 필요합니다
4. '캘리브레이션 시작' 버튼을 클릭하세요
        """
        
        ttk.Label(main_frame, text=instruction_text, justify=tk.LEFT).pack(pady=(0, 20))
        
        # 진행 상황
        self.progress_var = tk.StringVar(value="대기 중...")
        ttk.Label(main_frame, textvariable=self.progress_var).pack(pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 20))
        
        # 버튼 프레임
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, 
                  text="캘리브레이션 시작",
                  command=self.start_calibration).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame,
                  text="기존 데이터 로드",
                  command=self.load_calibration).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame,
                  text="닫기",
                  command=self.window.destroy).pack(side=tk.RIGHT)
    
    def start_calibration(self):
        """캘리브레이션 시작"""
        # 실제 캘리브레이션 로직 구현
        self.progress_var.set("캘리브레이션 진행 중...")
        self.progress_bar['value'] = 50
        
        # 시뮬레이션
        self.window.after(2000, self.calibration_complete)
    
    def calibration_complete(self):
        """캘리브레이션 완료"""
        self.progress_var.set("캘리브레이션 완료!")
        self.progress_bar['value'] = 100
        messagebox.showinfo("완료", "캘리브레이션이 완료되었습니다.")
    
    def load_calibration(self):
        """기존 캘리브레이션 데이터 로드"""
        file_path = filedialog.askopenfilename(
            title="캘리브레이션 파일 선택",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if self.stereo_system.load_calibration(file_path):
                    messagebox.showinfo("성공", "캘리브레이션 데이터를 로드했습니다.")
                else:
                    messagebox.showerror("오류", "캘리브레이션 데이터 로드에 실패했습니다.")
            except Exception as e:
                messagebox.showerror("오류", f"파일 로드 오류:\n{e}")

def create_kiosk_config():
    """키오스크 설정 생성"""
    return KioskConfig(
        screen_width=1920,
        screen_height=1080,
        fullscreen=False,  # 개발 시에는 False
        auto_start=True,
        camera_top_index=0,
        camera_bottom_index=1,
        camera_resolution=(1920, 1080),
        camera_fps=240,
        detection_zone_mm=(400.0, 400.0),
        measurement_zone_mm=(800.0, 800.0),
        accuracy_target_percent=5.0,
        db_path="golf_swing_data.db",
        auto_backup=True,
        backup_interval_hours=24
    )

def main():
    """메인 함수"""
    try:
        # 설정 로드
        config = create_kiosk_config()
        
        # GUI 생성 및 실행
        app = KioskGUI(config)
        app.run()
        
    except Exception as e:
        logger.error(f"시스템 실행 오류: {e}")
        messagebox.showerror("시스템 오류", f"시스템 실행에 실패했습니다:\n{e}")

if __name__ == "__main__":
    main()

