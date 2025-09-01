"""
Playwright 테스트 스크립트
브라우저 자동화 및 E2E 테스트 예제
"""

from playwright.sync_api import sync_playwright
import time

def test_playwright_installation():
    """Playwright 설치 및 작동 테스트"""
    
    with sync_playwright() as p:
        # 브라우저 실행 (headless 모드)
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # 웹 페이지 방문
        page.goto("https://www.google.com")
        
        # 페이지 제목 확인
        title = page.title()
        print(f"페이지 제목: {title}")
        
        # 스크린샷 저장
        page.screenshot(path="test_screenshot.png")
        print("스크린샷 저장 완료: test_screenshot.png")
        
        # 브라우저 종료
        browser.close()
        
        print("✅ Playwright 테스트 성공!")
        return True

def test_golf_web_dashboard():
    """골프 스윙 분석 웹 대시보드 테스트"""
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # GUI 모드로 실행
        page = browser.new_page()
        
        try:
            # 로컬 웹 대시보드 접속 시도
            page.goto("http://localhost:5000", timeout=5000)
            print("웹 대시보드 접속 성공")
            
            # 페이지 요소 확인
            if page.is_visible("text=Golf Swing Analysis"):
                print("대시보드 타이틀 확인")
            
        except Exception as e:
            print(f"웹 대시보드 접속 실패 (서버가 실행 중이 아닐 수 있음): {e}")
        
        finally:
            browser.close()

if __name__ == "__main__":
    print("=" * 50)
    print("Playwright 설치 테스트")
    print("=" * 50)
    
    try:
        # 기본 테스트 실행
        test_playwright_installation()
        
        print("\n" + "=" * 50)
        print("웹 대시보드 테스트 (선택적)")
        print("=" * 50)
        
        # 웹 대시보드 테스트 (실패해도 무방)
        test_golf_web_dashboard()
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print("Playwright가 제대로 설치되지 않았을 수 있습니다.")