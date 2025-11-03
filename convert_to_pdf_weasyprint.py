#!/usr/bin/env python3
"""
WeasyPrint를 사용하여 Markdown을 PDF로 변환하는 스크립트
"""

import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import os

def convert_md_to_pdf_weasyprint(md_file, pdf_file):
    """Markdown 파일을 PDF로 변환 (WeasyPrint 사용)"""
    
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
            @page {{
                size: A4;
                margin: 2cm;
            }}
            
            body {{
                font-family: 'Malgun Gothic', 'Arial', sans-serif;
                line-height: 1.6;
                color: #333;
                font-size: 12px;
            }}
            
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                page-break-after: avoid;
                font-size: 24px;
                margin-top: 0;
            }}
            
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 5px;
                margin-top: 30px;
                page-break-after: avoid;
                font-size: 18px;
            }}
            
            h3 {{
                color: #7f8c8d;
                margin-top: 25px;
                page-break-after: avoid;
                font-size: 16px;
            }}
            
            h4 {{
                color: #95a5a6;
                margin-top: 20px;
                page-break-after: avoid;
                font-size: 14px;
            }}
            
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                page-break-inside: avoid;
                font-size: 11px;
            }}
            
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                vertical-align: top;
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
                font-size: 10px;
            }}
            
            pre {{
                background-color: #f8f8f8;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
                page-break-inside: avoid;
                font-size: 10px;
            }}
            
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 20px 0;
                padding: 10px 20px;
                background-color: #f8f9fa;
                page-break-inside: avoid;
            }}
            
            ul, ol {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            
            li {{
                margin: 5px 0;
            }}
            
            .emoji {{
                font-size: 1.2em;
            }}
            
            .page-break {{
                page-break-before: always;
            }}
            
            .no-break {{
                page-break-inside: avoid;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    try:
        # WeasyPrint로 PDF 생성
        font_config = FontConfiguration()
        html_doc = HTML(string=html_with_style)
        html_doc.write_pdf(pdf_file, font_config=font_config)
        
        print(f"PDF 변환 완료: {pdf_file}")
        return True
    except Exception as e:
        print(f"PDF 변환 실패: {e}")
        return False

def main():
    """메인 함수"""
    md_file = "golf_ball_analysis_report.md"
    pdf_file = "golf_ball_analysis_report.pdf"
    
    if os.path.exists(md_file):
        success = convert_md_to_pdf_weasyprint(md_file, pdf_file)
        if success:
            print(f"\n보고서가 성공적으로 생성되었습니다:")
            print(f"   - Markdown: {md_file}")
            print(f"   - PDF: {pdf_file}")
        else:
            print(f"\nPDF 변환에 실패했습니다.")
    else:
        print(f"Markdown 파일을 찾을 수 없습니다: {md_file}")

if __name__ == "__main__":
    main()

