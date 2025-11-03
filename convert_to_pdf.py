#!/usr/bin/env python3
"""
Markdown을 PDF로 변환하는 스크립트
"""

import markdown
import pdfkit
import os

def convert_md_to_pdf(md_file, pdf_file):
    """Markdown 파일을 PDF로 변환"""
    
    # Markdown 파일 읽기
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Markdown을 HTML로 변환
    html = markdown.markdown(md_content, extensions=['tables', 'codehilite', 'fenced_code'])
    
    # HTML 헤더 추가
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Malgun Gothic', Arial, sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            h3 {{
                color: #7f8c8d;
                margin-top: 25px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f8f8f8;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 20px 0;
                padding: 10px 20px;
                background-color: #f8f9fa;
            }}
            .emoji {{
                font-size: 1.2em;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    # PDF 변환 옵션
    options = {
        'page-size': 'A4',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None
    }
    
    try:
        # PDF 생성
        pdfkit.from_string(html_with_style, pdf_file, options=options)
        print(f"✅ PDF 변환 완료: {pdf_file}")
        return True
    except Exception as e:
        print(f"❌ PDF 변환 실패: {e}")
        return False

def main():
    """메인 함수"""
    md_file = "golf_ball_analysis_report.md"
    pdf_file = "golf_ball_analysis_report.pdf"
    
    if os.path.exists(md_file):
        success = convert_md_to_pdf(md_file, pdf_file)
        if success:
            print(f"\n📄 보고서가 성공적으로 생성되었습니다:")
            print(f"   - Markdown: {md_file}")
            print(f"   - PDF: {pdf_file}")
        else:
            print(f"\n❌ PDF 변환에 실패했습니다.")
    else:
        print(f"❌ Markdown 파일을 찾을 수 없습니다: {md_file}")

if __name__ == "__main__":
    main()

