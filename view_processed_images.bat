@echo off
echo =====================================
echo 골프 스윙 이미지 뷰어
echo =====================================
echo.
echo 1. v7.0 최종 처리된 이미지 (권장)
echo 2. v7.0 기본 처리된 이미지  
echo 3. BMP 처리된 이미지
echo 4. 직접 경로 입력
echo 5. IrfanView로 폴더 열기
echo.
set /p choice="선택 (1-5): "

if "%choice%"=="1" (
    python view_images.py "C:\src\GolfSwingAnalysis_Final_ver8\data\images\shot-image-improved-v7-final\driver\no_marker_ball-1"
) else if "%choice%"=="2" (
    python view_images.py "C:\src\GolfSwingAnalysis_Final_ver8\data\images\shot-image-improved-v7\driver\no_marker_ball-1"
) else if "%choice%"=="3" (
    python view_images.py "C:\src\GolfSwingAnalysis_Final_ver8\data\images\shot-image-bmp-treated-3\driver\no_marker_ball-1"
) else if "%choice%"=="4" (
    set /p path="이미지 파일 또는 폴더 경로: "
    python view_images.py "%path%"
) else if "%choice%"=="5" (
    start "" "C:\Program Files\IrfanView\i_view64.exe" "C:\src\GolfSwingAnalysis_Final_ver8\data\images\shot-image-improved-v7-final\driver\no_marker_ball-1"
) else (
    echo 잘못된 선택입니다.
)

echo.
pause