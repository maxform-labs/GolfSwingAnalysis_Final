"""
골프 스윙 분석 웹 대시보드
Author: Maxform 개발팀
Description: 실시간 골프 스윙 분석 결과 웹 대시보드
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import sqlite3
import json
import os
import time
from datetime import datetime, timedelta
import threading
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebDashboard:
    """웹 대시보드 시스템"""
    
    def __init__(self, db_path: str = "golf_swing_data.db", port: int = 5000):
        """
        웹 대시보드 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
            port: 웹 서버 포트
        """
        self.db_path = db_path
        self.port = port
        
        # Flask 앱 설정
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'golf_swing_analysis_2025'
        
        # SocketIO 설정 (실시간 통신)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # 라우트 설정
        self.setup_routes()
        self.setup_socketio_events()
        
        # 실시간 데이터 스레드
        self.realtime_thread = None
        self.is_running = False
        
        logger.info(f"웹 대시보드 초기화 완료 - ver.2 (2025-01-29)")
    
    def setup_routes(self):
        """라우트 설정"""
        
        @self.app.route('/')
        def index():
            """메인 대시보드"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/swing_data')
        def get_swing_data():
            """스윙 데이터 API"""
            limit = request.args.get('limit', 10, type=int)
            data = self.get_recent_swings(limit)
            return jsonify(data)
        
        @self.app.route('/api/statistics')
        def get_statistics():
            """통계 데이터 API"""
            days = request.args.get('days', 7, type=int)
            stats = self.calculate_statistics(days)
            return jsonify(stats)
        
        @self.app.route('/api/performance_chart')
        def get_performance_chart():
            """성능 차트 API"""
            chart_type = request.args.get('type', 'ball_speed')
            days = request.args.get('days', 7, type=int)
            chart_data = self.generate_performance_chart(chart_type, days)
            return jsonify(chart_data)
        
        @self.app.route('/api/export_data')
        def export_data():
            """데이터 내보내기 API"""
            format_type = request.args.get('format', 'csv')
            days = request.args.get('days', 30, type=int)
            
            if format_type == 'csv':
                file_path = self.export_to_csv(days)
                return send_file(file_path, as_attachment=True)
            elif format_type == 'json':
                file_path = self.export_to_json(days)
                return send_file(file_path, as_attachment=True)
            else:
                return jsonify({'error': 'Unsupported format'}), 400
        
        @self.app.route('/calibration')
        def calibration_page():
            """캘리브레이션 페이지"""
            return render_template('calibration.html')
        
        @self.app.route('/settings')
        def settings_page():
            """설정 페이지"""
            return render_template('settings.html')
        
        @self.app.route('/api/system_status')
        def get_system_status():
            """시스템 상태 API"""
            status = self.get_system_status()
            return jsonify(status)
    
    def setup_socketio_events(self):
        """SocketIO 이벤트 설정"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """클라이언트 연결"""
            logger.info("클라이언트 연결됨")
            emit('status', {'message': '연결 성공'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """클라이언트 연결 해제"""
            logger.info("클라이언트 연결 해제됨")
        
        @self.socketio.on('request_realtime_data')
        def handle_realtime_request():
            """실시간 데이터 요청"""
            if not self.is_running:
                self.start_realtime_updates()
        
        @self.socketio.on('stop_realtime_data')
        def handle_stop_realtime():
            """실시간 데이터 중지"""
            self.stop_realtime_updates()
    
    def get_recent_swings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 스윙 데이터 조회"""
        try:
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
                
        except Exception as e:
            logger.error(f"스윙 데이터 조회 오류: {e}")
            return []
    
    def calculate_statistics(self, days: int = 7) -> Dict[str, Any]:
        """통계 계산"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 날짜 범위 설정
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_swings,
                        AVG(ball_speed_ms) as avg_ball_speed,
                        AVG(club_speed_ms) as avg_club_speed,
                        AVG(launch_angle_deg) as avg_launch_angle,
                        AVG(accuracy_score) as avg_accuracy,
                        MAX(ball_speed_ms) as max_ball_speed,
                        MIN(ball_speed_ms) as min_ball_speed
                    FROM swing_data 
                    WHERE timestamp >= ?
                ''', (start_date.isoformat(),))
                
                result = cursor.fetchone()
                
                if result:
                    stats = {
                        'total_swings': result[0] or 0,
                        'avg_ball_speed': round(result[1] or 0, 1),
                        'avg_club_speed': round(result[2] or 0, 1),
                        'avg_launch_angle': round(result[3] or 0, 1),
                        'avg_accuracy': round(result[4] or 0, 1),
                        'max_ball_speed': round(result[5] or 0, 1),
                        'min_ball_speed': round(result[6] or 0, 1),
                        'period_days': days
                    }
                    
                    # 개선 추세 계산
                    stats['improvement_trend'] = self.calculate_improvement_trend(days)
                    
                    return stats
                else:
                    return {'total_swings': 0, 'period_days': days}
                    
        except Exception as e:
            logger.error(f"통계 계산 오류: {e}")
            return {'error': str(e)}
    
    def calculate_improvement_trend(self, days: int) -> Dict[str, float]:
        """개선 추세 계산"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 전반부와 후반부 비교
                end_date = datetime.now()
                mid_date = end_date - timedelta(days=days//2)
                start_date = end_date - timedelta(days=days)
                
                # 전반부 데이터
                cursor.execute('''
                    SELECT AVG(ball_speed_ms), AVG(accuracy_score)
                    FROM swing_data 
                    WHERE timestamp BETWEEN ? AND ?
                ''', (start_date.isoformat(), mid_date.isoformat()))
                
                first_half = cursor.fetchone()
                
                # 후반부 데이터
                cursor.execute('''
                    SELECT AVG(ball_speed_ms), AVG(accuracy_score)
                    FROM swing_data 
                    WHERE timestamp >= ?
                ''', (mid_date.isoformat(),))
                
                second_half = cursor.fetchone()
                
                if first_half and second_half and first_half[0] and second_half[0]:
                    speed_trend = ((second_half[0] - first_half[0]) / first_half[0]) * 100
                    accuracy_trend = ((second_half[1] - first_half[1]) / first_half[1]) * 100
                    
                    return {
                        'speed_improvement': round(speed_trend, 1),
                        'accuracy_improvement': round(accuracy_trend, 1)
                    }
                else:
                    return {'speed_improvement': 0, 'accuracy_improvement': 0}
                    
        except Exception as e:
            logger.error(f"개선 추세 계산 오류: {e}")
            return {'speed_improvement': 0, 'accuracy_improvement': 0}
    
    def generate_performance_chart(self, chart_type: str, days: int) -> Dict[str, Any]:
        """성능 차트 생성"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # 데이터 조회
                cursor.execute(f'''
                    SELECT timestamp, {chart_type}
                    FROM swing_data 
                    WHERE timestamp >= ? AND {chart_type} IS NOT NULL
                    ORDER BY timestamp
                ''', (start_date.isoformat(),))
                
                data = cursor.fetchall()
                
                if not data:
                    return {'labels': [], 'values': [], 'chart_type': chart_type}
                
                # 데이터 처리
                timestamps = [datetime.fromisoformat(row[0]) for row in data]
                values = [row[1] for row in data]
                
                # 날짜별 평균 계산
                daily_data = {}
                for ts, val in zip(timestamps, values):
                    date_key = ts.date().isoformat()
                    if date_key not in daily_data:
                        daily_data[date_key] = []
                    daily_data[date_key].append(val)
                
                # 평균값 계산
                labels = sorted(daily_data.keys())
                avg_values = [np.mean(daily_data[date]) for date in labels]
                
                return {
                    'labels': labels,
                    'values': avg_values,
                    'chart_type': chart_type,
                    'unit': self.get_unit_for_metric(chart_type)
                }
                
        except Exception as e:
            logger.error(f"성능 차트 생성 오류: {e}")
            return {'error': str(e)}
    
    def get_unit_for_metric(self, metric: str) -> str:
        """메트릭에 대한 단위 반환"""
        if metric.endswith('_ms'):
            return 'm/s'
        elif metric.endswith('_deg'):
            return '°'
        elif metric.endswith('_rpm'):
            return 'rpm'
        elif metric == 'accuracy_score':
            return '%'
        else:
            return ''
    
    def export_to_csv(self, days: int) -> str:
        """CSV로 데이터 내보내기"""
        try:
            import pandas as pd
            
            with sqlite3.connect(self.db_path) as conn:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                query = '''
                    SELECT * FROM swing_data 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                '''
                
                df = pd.read_sql_query(query, conn, params=(start_date.isoformat(),))
                
                # 파일 저장
                filename = f"golf_swing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                filepath = os.path.join('exports', filename)
                
                # 디렉토리 생성
                os.makedirs('exports', exist_ok=True)
                
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
                
                return filepath
                
        except Exception as e:
            logger.error(f"CSV 내보내기 오류: {e}")
            raise
    
    def export_to_json(self, days: int) -> str:
        """JSON으로 데이터 내보내기"""
        try:
            data = self.get_recent_swings(1000)  # 충분한 데이터 조회
            
            # 날짜 필터링
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            filtered_data = []
            for item in data:
                item_date = datetime.fromisoformat(item['timestamp'])
                if item_date >= start_date:
                    filtered_data.append(item)
            
            # 파일 저장
            filename = f"golf_swing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join('exports', filename)
            
            # 디렉토리 생성
            os.makedirs('exports', exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"JSON 내보내기 오류: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            # 데이터베이스 상태
            db_status = os.path.exists(self.db_path)
            
            # 최근 활동
            recent_swings = self.get_recent_swings(1)
            last_activity = None
            if recent_swings:
                last_activity = recent_swings[0]['timestamp']
            
            # 디스크 사용량
            disk_usage = self.get_disk_usage()
            
            return {
                'database_connected': db_status,
                'last_activity': last_activity,
                'disk_usage_mb': disk_usage,
                'server_time': datetime.now().isoformat(),
                'uptime_seconds': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
            
        except Exception as e:
            logger.error(f"시스템 상태 조회 오류: {e}")
            return {'error': str(e)}
    
    def get_disk_usage(self) -> float:
        """디스크 사용량 조회 (MB)"""
        try:
            if os.path.exists(self.db_path):
                return os.path.getsize(self.db_path) / (1024 * 1024)
            return 0
        except:
            return 0
    
    def start_realtime_updates(self):
        """실시간 업데이트 시작"""
        if not self.is_running:
            self.is_running = True
            self.realtime_thread = threading.Thread(target=self.realtime_worker, daemon=True)
            self.realtime_thread.start()
            logger.info("실시간 업데이트 시작")
    
    def stop_realtime_updates(self):
        """실시간 업데이트 중지"""
        self.is_running = False
        logger.info("실시간 업데이트 중지")
    
    def realtime_worker(self):
        """실시간 데이터 워커"""
        while self.is_running:
            try:
                # 최신 데이터 조회
                recent_data = self.get_recent_swings(1)
                if recent_data:
                    self.socketio.emit('realtime_data', recent_data[0])
                
                # 시스템 상태 전송
                status = self.get_system_status()
                self.socketio.emit('system_status', status)
                
                time.sleep(1)  # 1초마다 업데이트
                
            except Exception as e:
                logger.error(f"실시간 워커 오류: {e}")
                time.sleep(5)
    
    def create_templates(self):
        """HTML 템플릿 생성"""
        # 템플릿 디렉토리 생성
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static/css', exist_ok=True)
        os.makedirs('static/js', exist_ok=True)
        
        # 메인 대시보드 템플릿
        dashboard_html = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>골프 스윙 분석 대시보드</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <style>
        .dashboard-card {
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-online { background-color: #28a745; }
        .status-offline { background-color: #dc3545; }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">
                <i class="fas fa-golf-ball"></i> 골프 스윙 분석 대시보드
            </span>
            <div class="d-flex">
                <span class="navbar-text me-3">
                    <span id="status-indicator" class="status-indicator status-offline"></span>
                    <span id="connection-status">연결 중...</span>
                </span>
                <button class="btn btn-outline-light btn-sm" onclick="exportData('csv')">
                    <i class="fas fa-download"></i> CSV 내보내기
                </button>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- 메트릭 카드들 -->
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value" id="total-swings">0</div>
                    <div class="metric-label">총 스윙 수</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value" id="avg-ball-speed">0</div>
                    <div class="metric-label">평균 볼 스피드 (m/s)</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value" id="avg-accuracy">0</div>
                    <div class="metric-label">평균 정확도 (%)</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value" id="max-ball-speed">0</div>
                    <div class="metric-label">최고 볼 스피드 (m/s)</div>
                </div>
            </div>
        </div>

        <div class="row">
            <!-- 실시간 데이터 -->
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-line"></i> 실시간 데이터</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="realtimeChart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>

            <!-- 최근 스윙 데이터 -->
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-header">
                        <h5><i class="fas fa-history"></i> 최근 스윙 데이터</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>시간</th>
                                        <th>볼 스피드</th>
                                        <th>발사각</th>
                                        <th>정확도</th>
                                    </tr>
                                </thead>
                                <tbody id="recent-swings">
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 성능 차트 -->
        <div class="row">
            <div class="col-12">
                <div class="card dashboard-card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-area"></i> 성능 추이</h5>
                        <div class="btn-group btn-group-sm" role="group">
                            <button type="button" class="btn btn-outline-primary" onclick="loadChart('ball_speed_ms')">볼 스피드</button>
                            <button type="button" class="btn btn-outline-primary" onclick="loadChart('accuracy_score')">정확도</button>
                            <button type="button" class="btn btn-outline-primary" onclick="loadChart('launch_angle_deg')">발사각</button>
                        </div>
                    </div>
                    <div class="card-body">
                        <canvas id="performanceChart" width="400" height="150"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Socket.IO 연결
        const socket = io();
        
        // 차트 초기화
        let realtimeChart, performanceChart;
        
        socket.on('connect', function() {
            document.getElementById('connection-status').textContent = '연결됨';
            document.getElementById('status-indicator').className = 'status-indicator status-online';
            socket.emit('request_realtime_data');
        });

        socket.on('disconnect', function() {
            document.getElementById('connection-status').textContent = '연결 끊김';
            document.getElementById('status-indicator').className = 'status-indicator status-offline';
        });

        socket.on('realtime_data', function(data) {
            updateRealtimeData(data);
        });

        // 초기 데이터 로드
        window.onload = function() {
            loadStatistics();
            loadRecentSwings();
            initCharts();
            loadChart('ball_speed_ms');
        };

        function loadStatistics() {
            fetch('/api/statistics')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-swings').textContent = data.total_swings || 0;
                    document.getElementById('avg-ball-speed').textContent = data.avg_ball_speed || 0;
                    document.getElementById('avg-accuracy').textContent = data.avg_accuracy || 0;
                    document.getElementById('max-ball-speed').textContent = data.max_ball_speed || 0;
                });
        }

        function loadRecentSwings() {
            fetch('/api/swing_data?limit=10')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('recent-swings');
                    tbody.innerHTML = '';
                    
                    data.forEach(swing => {
                        const row = tbody.insertRow();
                        const time = new Date(swing.timestamp).toLocaleTimeString();
                        row.innerHTML = `
                            <td>${time}</td>
                            <td>${swing.ball_speed_ms || 0} m/s</td>
                            <td>${swing.launch_angle_deg || 0}°</td>
                            <td>${swing.accuracy_score || 0}%</td>
                        `;
                    });
                });
        }

        function initCharts() {
            // 실시간 차트
            const realtimeCtx = document.getElementById('realtimeChart').getContext('2d');
            realtimeChart = new Chart(realtimeCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '볼 스피드 (m/s)',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { display: false }
                    }
                }
            });

            // 성능 차트
            const performanceCtx = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(performanceCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true
                }
            });
        }

        function updateRealtimeData(data) {
            if (realtimeChart && data.ball_speed_ms) {
                const now = new Date().toLocaleTimeString();
                
                realtimeChart.data.labels.push(now);
                realtimeChart.data.datasets[0].data.push(data.ball_speed_ms);
                
                // 최대 20개 데이터 포인트 유지
                if (realtimeChart.data.labels.length > 20) {
                    realtimeChart.data.labels.shift();
                    realtimeChart.data.datasets[0].data.shift();
                }
                
                realtimeChart.update();
            }
        }

        function loadChart(chartType) {
            fetch(`/api/performance_chart?type=${chartType}&days=7`)
                .then(response => response.json())
                .then(data => {
                    if (performanceChart) {
                        performanceChart.data.labels = data.labels;
                        performanceChart.data.datasets[0].data = data.values;
                        performanceChart.data.datasets[0].label = `${chartType} (${data.unit})`;
                        performanceChart.update();
                    }
                });
        }

        function exportData(format) {
            window.open(`/api/export_data?format=${format}&days=30`, '_blank');
        }

        // 주기적 업데이트
        setInterval(loadStatistics, 30000); // 30초마다
        setInterval(loadRecentSwings, 10000); // 10초마다
    </script>
</body>
</html>
        '''
        
        with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        logger.info("HTML 템플릿 생성 완료")
    
    def run(self, debug: bool = False):
        """웹 서버 실행"""
        self.start_time = time.time()
        self.create_templates()
        
        logger.info(f"웹 대시보드 서버 시작: http://localhost:{self.port}")
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=debug)

def create_web_dashboard():
    """웹 대시보드 생성"""
    return WebDashboard(
        db_path="golf_swing_data.db",
        port=5000
    )

if __name__ == "__main__":
    # 웹 대시보드 실행
    dashboard = create_web_dashboard()
    dashboard.run(debug=True)

