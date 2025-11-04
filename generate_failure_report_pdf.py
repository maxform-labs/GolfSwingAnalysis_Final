"""
실패 분석 보고서 PDF 생성기
이미지를 적절한 위치에 삽입하여 최종 PDF 생성
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, 
                                PageBreak, Image, Table, TableStyle, KeepTogether)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

def setup_korean_fonts():
    """한글 폰트 설정"""
    try:
        # Windows 기본 한글 폰트
        pdfmetrics.registerFont(TTFont('NanumGothic', 'c:/Windows/Fonts/malgun.ttf'))
        pdfmetrics.registerFont(TTFont('NanumGothic-Bold', 'c:/Windows/Fonts/malgunbd.ttf'))
        return True
    except:
        try:
            pdfmetrics.registerFont(TTFont('NanumGothic', 'c:/Windows/Fonts/NanumGothic.ttf'))
            pdfmetrics.registerFont(TTFont('NanumGothic-Bold', 'c:/Windows/Fonts/NanumGothicBold.ttf'))
            return True
        except:
            print("Warning: 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            return False

def create_styles(has_korean_font):
    """스타일 생성"""
    styles = getSampleStyleSheet()
    font_name = 'NanumGothic' if has_korean_font else 'Helvetica'
    font_bold = 'NanumGothic-Bold' if has_korean_font else 'Helvetica-Bold'
    
    # 제목 스타일
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontName=font_bold,
        fontSize=24,
        textColor=HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER
    ))
    
    # 섹션 제목
    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading1'],
        fontName=font_bold,
        fontSize=16,
        textColor=HexColor('#d32f2f'),
        spaceAfter=12,
        spaceBefore=20
    ))
    
    # 서브섹션 제목
    styles.add(ParagraphStyle(
        name='SubsectionTitle',
        parent=styles['Heading2'],
        fontName=font_bold,
        fontSize=14,
        textColor=HexColor('#1976d2'),
        spaceAfter=10,
        spaceBefore=15
    ))
    
    # 본문
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['BodyText'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    ))
    
    # 코드 블록
    styles.add(ParagraphStyle(
        name='CodeBlock',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        leftIndent=20,
        rightIndent=20,
        spaceBefore=6,
        spaceAfter=6,
        backColor=HexColor('#f5f5f5')
    ))
    
    # 강조
    styles.add(ParagraphStyle(
        name='Emphasis',
        parent=styles['CustomBody'],
        fontName=font_bold,
        textColor=HexColor('#d32f2f')
    ))
    
    # 이미지 캡션
    styles.add(ParagraphStyle(
        name='Caption',
        parent=styles['CustomBody'],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=HexColor('#666666'),
        spaceBefore=5,
        spaceAfter=10
    ))
    
    return styles

def add_image_with_caption(story, image_path, caption, width=15*cm):
    """이미지와 캡션 추가"""
    if os.path.exists(image_path):
        img = Image(image_path, width=width, height=width*0.6, kind='proportional')
        story.append(img)
        story.append(Paragraph(f"<i>{caption}</i>", styles['Caption']))
        story.append(Spacer(1, 0.3*cm))
    else:
        print(f"Warning: 이미지를 찾을 수 없습니다: {image_path}")

def create_failure_report_pdf():
    """실패 분석 보고서 PDF 생성"""
    
    global styles
    has_korean_font = setup_korean_fonts()
    styles = create_styles(has_korean_font)
    
    # PDF 설정
    pdf_filename = "골프공_3D_추적_시스템_실패_분석_보고서.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    story = []
    
    # ==================== 표지 ====================
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("골프공 3D 추적 시스템", styles['CustomTitle']))
    story.append(Paragraph("실패 분석 보고서", styles['CustomTitle']))
    story.append(Spacer(1, 1*cm))
    
    # Executive Summary
    summary_data = [
        ["작업 기간", "2025년 10월 30일 - 11월 3일"],
        ["목표", "수직 스테레오 비전 기반 골프공 3D 추적 시스템 정확도 향상"],
        ["방향각 오차", "❌ 94.55° (실용 불가)"],
        ["속도 오차", "❌ 51.2% (실용 불가)"],
        ["검출률", "⚠️ 62% (목표: 90%+)"],
        ["깊이 보정", "✅ 5.2% 오차 (유일한 성공)"],
    ]
    
    summary_table = Table(summary_data, colWidths=[5*cm, 10*cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#e3f2fd')),
        ('BACKGROUND', (1, 0), (1, -1), HexColor('#ffffff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'NanumGothic-Bold' if has_korean_font else 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'NanumGothic' if has_korean_font else 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 1*cm))
    
    story.append(Paragraph(
        "<b>핵심 결론:</b> 시스템은 실패했습니다. 수많은 개선 시도에도 불구하고 "
        "골프공의 움직임을 정확히 측정하지 못했습니다.",
        styles['Emphasis']
    ))
    
    story.append(PageBreak())
    
    # ==================== Section 1: 초기 문제 발견 ====================
    story.append(Paragraph("1. 시스템 초기 상태와 첫 번째 문제 발견", styles['SectionTitle']))
    
    story.append(Paragraph("1.1 하드웨어 구성", styles['SubsectionTitle']))
    
    hardware_text = """
    • 카메라: 고속 카메라 820 fps<br/>
    • 배치: 수직 스테레오 (위아래 470mm 간격)<br/>
    • 해상도: 1440×1080 (전체), 1440×300 (ROI 측정 영역)<br/>
    • 초점 거리: 1500 픽셀<br/>
    • 물리적 거리: 카메라1-공 900-1000mm, 카메라2-공 500-600mm
    """
    story.append(Paragraph(hardware_text, styles['CustomBody']))
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph("1.2 최초 테스트 결과 - 심각한 문제 발견", styles['SubsectionTitle']))
    
    initial_problems = [
        ["문제", "기대값", "계산값", "오차"],
        ["깊이 계산", "700-950mm", "9286mm", "20-40배"],
        ["속도 계산", "54-63 m/s", "174 m/s", "2.9배"],
        ["방향각", "-7° ~ 13°", "-170° ~ 180°", "무작위"],
    ]
    
    problem_table = Table(initial_problems, colWidths=[3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    problem_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#d32f2f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ffebee')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'NanumGothic-Bold' if has_korean_font else 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(problem_table)
    story.append(Spacer(1, 0.5*cm))
    
    # 분석 결과 그래프 삽입
    story.append(Paragraph("1.3 분석 결과 시각화", styles['SubsectionTitle']))
    
    # Speed Analysis
    speed_img_path = "driver_2_debug_result/speed_analysis.png"
    if os.path.exists(speed_img_path):
        add_image_with_caption(
            story, 
            speed_img_path,
            "그림 1: 속도 계산 오차 - 평균 51.2%, 일부 샷은 100% 이상 오차",
            width=14*cm
        )
    
    story.append(PageBreak())
    
    # Angle Analysis
    angle_img_path = "driver_2_debug_result/angle_analysis.png"
    if os.path.exists(angle_img_path):
        add_image_with_caption(
            story,
            angle_img_path,
            "그림 2: 각도 계산 오차 - 방향각 94.55° 평균 오차 (치명적)",
            width=14*cm
        )
    
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph(
        "<b>첫 번째 의문:</b> 캘리브레이션이 제대로 되었는가?",
        styles['Emphasis']
    ))
    
    story.append(PageBreak())
    
    # ==================== Section 2: 캘리브레이션 검증 ====================
    story.append(Paragraph("2. 첫 번째 시도: 캘리브레이션 파일 검증", styles['SectionTitle']))
    
    story.append(Paragraph("2.1 캘리브레이션 데이터 확인", styles['SubsectionTitle']))
    
    calib_findings = """
    <b>발견 1: 체스보드 이미지 부재</b><br/>
    캘리브레이션에 사용된 원본 체스보드 이미지가 없음 → 검증 불가능, 재캘리브레이션 불가능<br/><br/>
    
    <b>발견 2: 의심스러운 캘리브레이션 파라미터</b><br/>
    • 회전 행렬 R = Identity (단위 행렬) → 현실적으로 불가능<br/>
    • 왜곡 계수 D = [0, 0, 0, 0, 0] → 모든 렌즈에는 왜곡 존재<br/>
    • 초점 거리 f = 1500 픽셀 → 실측인지 가정인지 불명<br/><br/>
    
    <b>발견 3: ROI와 캘리브레이션 불일치</b><br/>
    • 캘리브레이션: 1440×1080 (전체 이미지)<br/>
    • 실제 측정: 1440×300 (ROI)<br/>
    • 주점 cy=540, ROI 중앙 y=546 → 6픽셀 차이가 체계적 바이어스 발생
    """
    story.append(Paragraph(calib_findings, styles['CustomBody']))
    story.append(Spacer(1, 0.5*cm))
    
    # 캘리브레이션 이미지 삽입
    story.append(Paragraph("2.2 체스보드 이미지 분석", styles['SubsectionTitle']))
    
    cam1_img = "driver_2_debug_result/cam1_chessboard_corners.jpg"
    cam2_img = "driver_2_debug_result/cam2_chessboard_corners.jpg"
    
    if os.path.exists(cam1_img):
        add_image_with_caption(
            story,
            cam1_img,
            "그림 3: Camera1 체스보드 코너 검출 - 검출된 코너 위치 (색상 표시)",
            width=13*cm
        )
    
    if os.path.exists(cam2_img):
        add_image_with_caption(
            story,
            cam2_img,
            "그림 4: Camera2 체스보드 코너 검출 - 두 카메라 간 대응점 매칭 필요",
            width=13*cm
        )
    
    story.append(PageBreak())
    
    story.append(Paragraph("2.3 깊이 보정 시도", styles['SubsectionTitle']))
    
    depth_correction_text = """
    <b>문제:</b> 깊이 계산이 20-40배 과대 추정<br/>
    <b>시도:</b> Grid search로 최적 scale factor 찾기 [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]<br/>
    <b>결과:</b> scale = 1.0이 최적 (샷 1에서 5.2% 오차 달성)<br/><br/>
    
    <b>문제점:</b><br/>
    • 왜 scale=1.0이 작동하는지 이론적 설명 불가<br/>
    • 다른 샷에서는 여전히 큰 오차 (일부 50% 이상)<br/>
    • 방향각은 전혀 개선 안 됨<br/><br/>
    
    <b>결론:</b> 증상은 완화했으나 근본 원인(캘리브레이션 부정확, 좌표계 불일치)은 미해결
    """
    story.append(Paragraph(depth_correction_text, styles['CustomBody']))
    
    story.append(PageBreak())
    
    # ==================== Section 3: 검출 개선 ====================
    story.append(Paragraph("3. 두 번째 시도: 볼 검출 개선", styles['SectionTitle']))
    
    story.append(Paragraph("3.1 검출 실패 진단", styles['SubsectionTitle']))
    
    detection_text = """
    <b>낮은 검출률:</b> 평균 62% (14/23 프레임)<br/>
    • 최고: 19 프레임 (샷 1, 17, 19)<br/>
    • 최저: 8 프레임 (샷 2)<br/><br/>
    
    <b>왜 검출이 어려운가?</b><br/>
    1. 고속 움직임: 60 m/s = 60,000 mm/s<br/>
    2. 프레임당 이동: 73.2 mm (볼 직경보다 큼)<br/>
    3. 심한 모션 블러 발생<br/>
    4. 볼 크기: 10-30 픽셀 (매우 작음)<br/>
    5. 배경 노이즈: 잔디, 그림자, 조명 변화<br/><br/>
    
    <b>시도한 방법:</b><br/>
    • Multi-scale 검출 (Adaptive Threshold + Hough Circle + Contour)<br/>
    • 파라미터 튜닝 (threshold 200→80, area 30-15000→20-20000)<br/>
    • Temporal Continuity (이전 프레임 위치 활용)<br/><br/>
    
    <b>결과:</b> 50% → 62% (12% 향상, 하지만 60%대 벽을 넘지 못함)
    """
    story.append(Paragraph(detection_text, styles['CustomBody']))
    story.append(Spacer(1, 0.5*cm))
    
    # 골프공 검출 샘플 이미지
    ball_samples = [
        "driver_2_debug_result/driver_2_shot_1_frame_00012.png",
        "driver_2_debug_result/driver_2_shot_1_frame_00015.png"
    ]
    
    for i, sample_path in enumerate(ball_samples, 1):
        if os.path.exists(sample_path):
            add_image_with_caption(
                story,
                sample_path,
                f"그림 {4+i}: 골프공 검출 샘플 - 녹색 원: 검출 성공, 빨간 점: 중심",
                width=12*cm
            )
    
    story.append(PageBreak())
    
    # ==================== Section 4: 칼만 필터 ====================
    story.append(Paragraph("4. 세 번째 시도: 칼만 필터 적용", styles['SectionTitle']))
    
    kalman_text = """
    <b>가설:</b> 검출 좌표가 불안정 → 칼만 필터로 스무딩 → 정확한 물리량<br/><br/>
    
    <b>구현:</b><br/>
    • 6-state 모델: [x, y, z, vx, vy, vz]<br/>
    • Q = 0.01 (프로세스 노이즈, 모델 신뢰)<br/>
    • R = 100 (측정 노이즈, 측정 불신)<br/><br/>
    
    <b>왜 실패했나?</b><br/>
    칼만 필터의 전제 조건:<br/>
    1. 측정값이 존재해야 함<br/>
    2. 측정 노이즈가 가우시안 분포<br/>
    3. 모델(등속도)이 실제와 유사<br/><br/>
    
    우리의 현실:<br/>
    1. 측정값 38% 누락 (검출률 62%)<br/>
    2. 검출 오류는 non-Gaussian (완전히 틀린 위치)<br/>
    3. 골프공은 감속함 (공기저항, 등속도 아님)<br/><br/>
    
    → <b>전제 조건이 모두 위반됨!</b><br/><br/>
    
    <b>결론:</b> 칼만 필터는 완전한 데이터의 노이즈를 제거하는 도구입니다. 
    불완전한 데이터(38% 누락)에는 무력합니다.
    """
    story.append(Paragraph(kalman_text, styles['CustomBody']))
    
    story.append(PageBreak())
    
    # ==================== Section 5: 좌표계 변환 ====================
    story.append(Paragraph("5. 네 번째 시도: 좌표계 변환 재정의", styles['SectionTitle']))
    
    coord_text = """
    <b>방향각 재앙:</b><br/>
    • 실제 방향각: -7° ~ 13° (20° 범위)<br/>
    • 계산 방향각: -170° ~ 180° (350° 범위)<br/>
    • 평균 오차: 94.55°<br/>
    • VZ 부호: 50% 양수, 50% 음수 (예상: 모두 양수)<br/><br/>
    
    <b>7가지 변환 행렬 테스트:</b><br/>
    """
    story.append(Paragraph(coord_text, styles['CustomBody']))
    
    transform_results = [
        ["변환", "평균 오차", "비고"],
        ["Identity", "76.15°", "기준"],
        ["Z-inversion", "98.01°", "더 나쁨"],
        ["X-Z swap", "111.63°", "더 나쁨"],
        ["X-Z swap + Z-inv", "112.35°", "더 나쁨"],
        ["X-Z swap + X-inv", "67.65°", "⭐ Best"],
        ["X-inversion", "81.99°", ""],
        ["X-inv + Z-inv", "103.85°", ""],
    ]
    
    transform_table = Table(transform_results, colWidths=[5*cm, 4*cm, 5*cm])
    transform_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1976d2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 4), (-1, 4), HexColor('#fff3e0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'NanumGothic-Bold' if has_korean_font else 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    
    story.append(transform_table)
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph(
        "<b>최적 변환 적용 결과:</b><br/>"
        "X_golf = -Z_cam, Y_golf = Y_cam, Z_golf = X_cam<br/>"
        "방향각 오차: 94.55° → 83.76° (11.4% 개선)<br/><br/>"
        "<b>문제:</b> 여전히 83.76°는 엄청난 오차 (목표: < 15°)<br/>"
        "좌표 변환은 올바른 축 매핑만 합니다. 잘못된 좌표 값 자체는 고칠 수 없습니다.",
        styles['CustomBody']
    ))
    
    story.append(PageBreak())
    
    # ==================== Section 6: Outlier 필터링 ====================
    story.append(Paragraph("6. 다섯 번째 시도: Outlier 필터링", styles['SectionTitle']))
    
    outlier_text = """
    <b>가설:</b> 일부 프레임의 검출 오류가 전체를 망침 → 이상치 제거 → 정확한 궤적<br/><br/>
    
    <b>구현 기법:</b><br/>
    1. RANSAC 궤적 피팅 (2차 다항식, threshold=100mm)<br/>
    2. Median Filter (window=5)<br/>
    3. Statistical Z-score (threshold=2.5σ)<br/><br/>
    
    <b>8개 문제 샷 테스트 결과:</b>
    """
    story.append(Paragraph(outlier_text, styles['CustomBody']))
    
    outlier_results = [
        ["샷", "실측", "원본 오차", "필터링 후", "결과"],
        ["1", "54.50 m/s", "5.2%", "55.2%", "❌ 악화"],
        ["2", "59.80 m/s", "-", "92.1%", "❌ 실패"],
        ["5", "56.50 m/s", "-", "31.2%", "⚠️ 약간 개선"],
        ["16", "55.50 m/s", "-", "100%", "❌❌ 완전 실패"],
        ["평균", "-", "51.2%", "63.4%", "❌ 악화"],
    ]
    
    outlier_table = Table(outlier_results, colWidths=[2*cm, 3*cm, 3*cm, 3*cm, 3.5*cm])
    outlier_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#d32f2f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, -1), (-1, -1), HexColor('#ffebee')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    
    story.append(outlier_table)
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph(
        "<b>왜 실패했나?</b><br/><br/>"
        "Outlier 필터링의 전제: [90% 좋은 데이터] + [10% 나쁜 데이터]<br/>"
        "우리의 현실: [62% 불확실한 데이터] + [38% 없는 데이터]<br/><br/>"
        "<b>교훈:</b> Outlier 필터링은 양질의 데이터에서 이상치를 제거하는 도구입니다. "
        "애초에 데이터가 부족하고 부정확하면 필터링이 상황을 더 악화시킵니다.",
        styles['Emphasis']
    ))
    
    story.append(PageBreak())
    
    # ==================== Section 7: 종합 분석 ====================
    story.append(Paragraph("7. 종합 실패 원인 분석", styles['SectionTitle']))
    
    story.append(Paragraph("7.1 연쇄 실패 구조", styles['SubsectionTitle']))
    
    failure_chain = """
    캘리브레이션 부정확<br/>
    ↓<br/>
    검출 좌표의 3D 변환 오류<br/>
    ↓<br/>
    검출률 저하 (62%) + 오검출<br/>
    ↓<br/>
    불완전한 궤적 데이터<br/>
    ↓<br/>
    시간 미분 시 오차 증폭<br/>
    ↓<br/>
    속도/방향각 계산 완전 실패<br/><br/>
    
    각 단계에서의 작은 오차가 누적되어 최종적으로는 사용 불가능한 결과를 만들어냈습니다.
    """
    story.append(Paragraph(failure_chain, styles['CustomBody']))
    
    story.append(Paragraph("7.2 실패 책임 할당", styles['SubsectionTitle']))
    
    responsibility = [
        ["원인", "책임도", "파급 효과"],
        ["캘리브레이션", "60%", "3D 좌표 부정확, Z축 방향 불명, 방향각 94.55° 오차"],
        ["검출 알고리즘", "30%", "38% 데이터 누락, 후속 처리 불안정"],
        ["좌표계 정의", "10%", "방향각 ±180° 오류, VZ 부호 혼란"],
    ]
    
    resp_table = Table(responsibility, colWidths=[4*cm, 3*cm, 7.5*cm])
    resp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#424242')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (0, 1), HexColor('#ffcdd2')),
        ('BACKGROUND', (0, 2), (0, 2), HexColor('#fff9c4')),
        ('BACKGROUND', (0, 3), (0, 3), HexColor('#c8e6c9')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(resp_table)
    
    story.append(PageBreak())
    
    # ==================== Section 8: 결론 ====================
    story.append(Paragraph("8. 결론: 실패로부터의 교훈", styles['SectionTitle']))
    
    story.append(Paragraph("8.1 4가지 핵심 교훈", styles['SubsectionTitle']))
    
    lessons = """
    <b>교훈 1: 기초가 전부다</b><br/>
    잘못된 캘리브레이션 위에 쌓은 모든 개선은 모래성입니다.<br/>
    <i>"Garbage In, Garbage Out"</i><br/>
    아무리 정교한 알고리즘도 잘못된 캘리브레이션은 고칠 수 없습니다.<br/><br/>
    
    <b>교훈 2: 데이터가 없으면 필터링도 없다</b><br/>
    검출률 62% = 38% 데이터 누락<br/>
    <i>"You can't filter what you don't have"</i><br/>
    필터링은 완전한 데이터의 노이즈를 제거하는 도구입니다.<br/><br/>
    
    <b>교훈 3: 증상 치료 vs 원인 치료</b><br/>
    우리는 증상을 10개 고쳤지만, 원인이 남아있어 여전히 실패했습니다.<br/>
    <i>"Treat the disease, not the symptoms"</i><br/><br/>
    
    <b>교훈 4: 알고리즘에는 한계가 있다</b><br/>
    <i>"No algorithm is magic"</i><br/>
    알고리즘은 특정 가정 하에서만 작동합니다. 가정이 위반되면 실패합니다.
    """
    story.append(Paragraph(lessons, styles['CustomBody']))
    
    story.append(PageBreak())
    
    story.append(Paragraph("8.2 최종 성과표", styles['SubsectionTitle']))
    
    final_results = [
        ["항목", "목표", "초기", "최종", "평가"],
        ["속도 오차", "< 20%", "38.9%", "51.2%", "❌ 악화"],
        ["발사각 오차", "< 8°", "8.12°", "9.41°", "❌ 악화"],
        ["방향각 오차", "< 15°", "75.35°", "83.76°", "❌ 목표 미달"],
        ["검출률", "> 90%", "50%", "62%", "❌ 목표 미달"],
    ]
    
    final_table = Table(final_results, colWidths=[3.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3.5*cm])
    final_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#424242')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ffebee')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    
    story.append(final_table)
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph(
        "<b>결론: 시스템은 실패했습니다</b>",
        styles['Emphasis']
    ))
    
    story.append(Spacer(1, 1*cm))
    
    story.append(Paragraph("8.3 만약 다시 한다면?", styles['SubsectionTitle']))
    
    restart_plan = """
    <b>Phase 1: 올바른 캘리브레이션 (필수, 1주)</b><br/>
    • 체스보드 패턴 100+ 이미지 쌍 촬영<br/>
    • OpenCV stereoCalibrate() 실행<br/>
    • R != Identity, D != 0 확인<br/>
    • Reprojection error < 0.5 픽셀 검증<br/><br/>
    
    <b>Phase 2: 딥러닝 검출기 YOLOv8 (필수, 1개월)</b><br/>
    • 920 이미지 데이터 준비 및 레이블링<br/>
    • YOLOv8 훈련 (100 epochs, GPU)<br/>
    • 예상 검출률: 90%+ 달성<br/><br/>
    
    <b>Phase 3: 시스템 검증 (1주)</b><br/>
    • 새로운 20샷 테스트<br/>
    • 합격 기준: 속도 < 20%, 방향각 < 15° 오차<br/><br/>
    
    <b>총 소요 시간: 6-8주</b><br/>
    <b>성공 확률: 80%+</b> (캘리브레이션 + 딥러닝)
    """
    story.append(Paragraph(restart_plan, styles['CustomBody']))
    
    story.append(PageBreak())
    
    story.append(Paragraph("8.4 마지막 말", styles['SubsectionTitle']))
    
    final_message = """
    이 5일간의 작업은 <b>실패</b>했습니다.<br/><br/>
    
    하지만 이 실패는 <b>가치 있는 실패</b>입니다:<br/>
    • 무엇이 작동하지 않는지 명확히 알게 되었습니다<br/>
    • 왜 작동하지 않는지 깊이 이해하게 되었습니다<br/>
    • 어떻게 고칠 수 있는지 구체적 방법을 제시했습니다<br/><br/>
    
    <b>캘리브레이션 없이는 아무것도 안 됩니다.</b><br/>
    <b>검출이 안 되면 계산도 안 됩니다.</b><br/>
    <b>기초가 흔들리면 모든 것이 무너집니다.</b><br/><br/>
    
    이제 선택은 명확합니다:<br/>
    1. 포기하거나<br/>
    2. 올바른 기초부터 다시 쌓거나<br/><br/>
    
    만약 2번을 선택한다면, 이 보고서가 <b>로드맵</b>이 될 것입니다.
    """
    story.append(Paragraph(final_message, styles['CustomBody']))
    
    story.append(Spacer(1, 2*cm))
    
    # 페이지 하단 정보
    footer_info = """
    <b>작성일:</b> 2025년 11월 3일<br/>
    <b>프로젝트:</b> Golf Swing Analysis (Vertical Stereo Vision)<br/>
    <b>버전:</b> 실패 분석 보고서 v2.0 (Failure Analysis Report)<br/>
    <b>상태:</b> 시스템 사용 불가 - 재시작 권장
    """
    story.append(Paragraph(footer_info, styles['Caption']))
    
    # PDF 생성
    doc.build(story)
    print(f"\n✅ PDF 생성 완료: {pdf_filename}")
    print(f"   파일 크기: {os.path.getsize(pdf_filename) / 1024:.1f} KB")
    
    return pdf_filename

if __name__ == "__main__":
    print("=" * 60)
    print("골프공 3D 추적 시스템 실패 분석 보고서 PDF 생성")
    print("=" * 60)
    
    pdf_file = create_failure_report_pdf()
    
    print("\n" + "=" * 60)
    print(f"생성된 파일: {pdf_file}")
    print("=" * 60)
