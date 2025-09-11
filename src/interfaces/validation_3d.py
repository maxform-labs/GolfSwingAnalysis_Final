#!/usr/bin/env python3
"""
현실적 95% 달성 3D 검증 앱
"""

import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template_string
import json

app = Flask(__name__)

# 데이터베이스 연결
def get_validation_data():
    conn = sqlite3.connect('./realistic_achievement_validation_results.db')
    df = pd.read_sql_query("SELECT * FROM realistic_validation_results", conn)
    conn.close()
    return df

@app.route('/')
def index():
    df = get_validation_data()
    
    # 3D 궤적 시각화 (드라이버 프로 데이터)
    pro_driver_data = df[(df['skill_level'] == '프로') & (df['club_type'] == '드라이버')]
    
    # 볼 스피드, 발사각, 방향각으로 3D 궤적 생성
    ball_speeds = pro_driver_data[pro_driver_data['parameter'] == 'ball_speed']['realistic_enhanced_value'].values[:10]
    launch_angles = pro_driver_data[pro_driver_data['parameter'] == 'launch_angle']['realistic_enhanced_value'].values[:10]
    
    # 3D 궤적 계산
    trajectories = []
    for i in range(min(len(ball_speeds), len(launch_angles))):
        speed = ball_speeds[i] if i < len(ball_speeds) else 171.0
        angle = launch_angles[i] if i < len(launch_angles) else 10.4
        
        # 간단한 포물선 궤적 계산
        t = np.linspace(0, 6, 100)
        x = speed * 0.44704 * np.cos(np.radians(angle)) * t  # mph to m/s
        y = np.zeros_like(t)  # 좌우 편차 없음
        z = speed * 0.44704 * np.sin(np.radians(angle)) * t - 0.5 * 9.81 * t**2
        
        # 지면에 닿으면 종료
        ground_idx = np.where(z < 0)[0]
        if len(ground_idx) > 0:
            end_idx = ground_idx[0]
            x = x[:end_idx]
            y = y[:end_idx]
            z = z[:end_idx]
        
        trajectories.append({'x': x.tolist(), 'y': y.tolist(), 'z': z.tolist()})
    
    # 파라미터별 정확도 차트
    param_accuracy = df.groupby('parameter').agg({
        'base_within_tolerance': 'mean',
        'realistic_enhanced_within_tolerance': 'mean'
    }).reset_index()
    
    param_accuracy['base_accuracy'] = param_accuracy['base_within_tolerance'] * 100
    param_accuracy['enhanced_accuracy'] = param_accuracy['realistic_enhanced_within_tolerance'] * 100
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Realistic 95% Achievement 3D Validation</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .chart { margin: 20px 0; }
            .stats { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .success { color: #28a745; font-weight: bold; }
            .warning { color: #ffc107; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏌️ Realistic Achievement System 검증</h1>
            
            <div class="stats">
                <h3>📊 전체 검증 결과</h3>
                <p><strong>총 테스트:</strong> {{ total_tests }}개</p>
                <p><strong>기존 정확도:</strong> {{ base_accuracy }}%</p>
                <p><strong>Realistic Enhanced 정확도:</strong> <span class="success">{{ enhanced_accuracy }}%</span></p>
                <p><strong>개선량:</strong> +{{ improvement }}%p</p>
                <p><strong>95% 목표 달성:</strong> <span class="{{ 'success' if target_achieved else 'warning' }}">{{ '성공' if target_achieved else '미달성' }}</span></p>
            </div>
            
            <div class="chart">
                <h3>🎯 3D 골프공 궤적 시뮬레이션 (프로 드라이버)</h3>
                <div id="trajectory3d"></div>
            </div>
            
            <div class="chart">
                <h3>📈 파라미터별 정확도 비교</h3>
                <div id="accuracy_chart"></div>
            </div>
            
            <div class="chart">
                <h3>🔄 실시간 측정 시뮬레이션</h3>
                <div id="realtime_chart"></div>
            </div>
        </div>
        
        <script>
            // 3D 궤적 차트
            var trajectories = {{ trajectories | safe }};
            var trajectory_data = [];
            
            for (var i = 0; i < trajectories.length; i++) {
                trajectory_data.push({
                    x: trajectories[i].x,
                    y: trajectories[i].y,
                    z: trajectories[i].z,
                    type: 'scatter3d',
                    mode: 'lines',
                    name: 'Shot ' + (i + 1),
                    line: { width: 3 }
                });
            }
            
            Plotly.newPlot('trajectory3d', trajectory_data, {
                title: '3D Golf Ball Trajectory (Professional Driver)',
                scene: {
                    xaxis: { title: 'Distance (m)' },
                    yaxis: { title: 'Side (m)' },
                    zaxis: { title: 'Height (m)' }
                },
                height: 500
            });
            
            // 정확도 비교 차트
            var param_data = {{ param_accuracy | safe }};
            var accuracy_data = [
                {
                    x: param_data.map(d => d.parameter),
                    y: param_data.map(d => d.base_accuracy),
                    type: 'bar',
                    name: '기존 시스템',
                    marker: { color: 'lightcoral' }
                },
                {
                    x: param_data.map(d => d.parameter),
                    y: param_data.map(d => d.enhanced_accuracy),
                    type: 'bar',
                    name: 'Realistic Enhanced',
                    marker: { color: 'lightblue' }
                }
            ];
            
            Plotly.newPlot('accuracy_chart', accuracy_data, {
                title: 'Parameter Accuracy Comparison',
                xaxis: { title: 'Parameters' },
                yaxis: { title: 'Accuracy (%)' },
                barmode: 'group',
                height: 400
            });
            
            // 실시간 시뮬레이션
            var realtime_data = [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Ball Speed (mph)',
                line: { color: 'blue' }
            }];
            
            var cnt = 0;
            function updateRealtime() {
                var new_speed = 171 + (Math.random() - 0.5) * 10;
                realtime_data[0].x.push(cnt);
                realtime_data[0].y.push(new_speed);
                
                if (realtime_data[0].x.length > 50) {
                    realtime_data[0].x.shift();
                    realtime_data[0].y.shift();
                }
                
                Plotly.redraw('realtime_chart');
                cnt++;
            }
            
            Plotly.newPlot('realtime_chart', realtime_data, {
                title: 'Real-time Ball Speed Measurement',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Speed (mph)' },
                height: 300
            });
            
            setInterval(updateRealtime, 100);
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html_template, 
                                trajectories=json.dumps(trajectories),
                                param_accuracy=param_accuracy.to_dict('records'),
                                total_tests=len(df),
                                base_accuracy=f"{df['base_within_tolerance'].mean() * 100:.2f}",
                                enhanced_accuracy=f"{df['realistic_enhanced_within_tolerance'].mean() * 100:.2f}",
                                improvement=f"{(df['realistic_enhanced_within_tolerance'].mean() - df['base_within_tolerance'].mean()) * 100:.2f}",
                                target_achieved=df['realistic_enhanced_within_tolerance'].mean() >= 0.95)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
