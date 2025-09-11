"""
BMP to JPG Converter for Golf Swing Analysis Project
Converts all BMP files from /shot-image to JPG format in /shot-image-jpg
Supports direct BMP processing for dimple analysis
"""

import os
from PIL import Image
from pathlib import Path
import shutil
import logging
from typing import List, Optional, Dict, Any

# 로컬 모듈 임포트
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.bmp_loader import create_bmp_loader

def convert_bmp_to_jpg(source_dir='shot-image', target_dir='shot-image-jpg'):
    """
    Convert all BMP files from source directory to JPG in target directory
    Maintains the same folder structure
    """
    
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(exist_ok=True)
    
    # Statistics
    total_files = 0
    converted_files = 0
    skipped_files = 0
    errors = []
    
    print(f"Starting BMP to JPG conversion...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print("-" * 50)
    
    # Walk through all directories and files
    for root, dirs, files in os.walk(source_dir):
        # Calculate relative path
        rel_path = os.path.relpath(root, source_dir)
        
        # Create corresponding directory in target
        if rel_path != '.':
            target_subdir = os.path.join(target_dir, rel_path)
        else:
            target_subdir = target_dir
            
        Path(target_subdir).mkdir(parents=True, exist_ok=True)
        
        # Process BMP files in current directory
        bmp_files = [f for f in files if f.lower().endswith('.bmp')]
        
        if bmp_files:
            print(f"\nProcessing {len(bmp_files)} files in: {rel_path}")
            
        for filename in bmp_files:
            total_files += 1
            source_file = os.path.join(root, filename)
            
            # Create target filename with .jpg extension
            target_filename = os.path.splitext(filename)[0] + '.jpg'
            target_file = os.path.join(target_subdir, target_filename)
            
            try:
                # Check if target already exists
                if os.path.exists(target_file):
                    print(f"  Skipping (already exists): {filename}")
                    skipped_files += 1
                    continue
                
                # Open and convert BMP to JPG
                with Image.open(source_file) as img:
                    # Convert RGBA to RGB if necessary
                    if img.mode in ('RGBA', 'LA', 'P'):
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as JPG with high quality
                    img.save(target_file, 'JPEG', quality=95, optimize=True)
                    converted_files += 1
                    
                    if converted_files % 50 == 0:
                        print(f"  Progress: {converted_files}/{total_files} files converted")
                        
            except Exception as e:
                error_msg = f"Error converting {source_file}: {str(e)}"
                print(f"  ERROR: {error_msg}")
                errors.append(error_msg)
    
    # Print summary
    print("\n" + "=" * 50)
    print("CONVERSION COMPLETE")
    print("=" * 50)
    print(f"Total BMP files found: {total_files}")
    print(f"Successfully converted: {converted_files}")
    print(f"Skipped (already exist): {skipped_files}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nError details:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    # Verify folder structure
    print("\n" + "-" * 50)
    print("Created folder structure:")
    for root, dirs, files in os.walk(target_dir):
        level = root.replace(target_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 2 * (level + 1)
        jpg_count = len([f for f in files if f.lower().endswith('.jpg')])
        if jpg_count > 0:
            print(f"{sub_indent}({jpg_count} JPG files)")
    
    return converted_files, skipped_files, errors

def analyze_bmp_directly(source_dir: str, club_type: str = 'driver') -> Dict[str, Any]:
    """
    BMP 파일을 직접 분석 (딤플 검출)
    
    Args:
        source_dir: BMP 파일이 있는 디렉토리
        club_type: 클럽 타입
        
    Returns:
        분석 결과 딕셔너리
    """
    logger = logging.getLogger(__name__)
    
    try:
        # BMP 로더 생성
        bmp_loader = create_bmp_loader(enable_cache=True)
        
        # BMP 파일 찾기
        bmp_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith('.bmp'):
                    bmp_files.append(os.path.join(root, file))
        
        if not bmp_files:
            logger.warning(f"BMP 파일을 찾을 수 없습니다: {source_dir}")
            return {"error": "BMP 파일을 찾을 수 없습니다"}
        
        # 파일 수 제한 (성능상 이유로)
        if len(bmp_files) > 20:
            bmp_files = bmp_files[:20]
            logger.info(f"처리할 파일 수를 20개로 제한했습니다.")
        
        # 분석기 임포트 및 실행
        from algorithms.spin_analysis.no_dimple_spin_final import FinalNoDimpleSpinAnalyzer
        
        analyzer = FinalNoDimpleSpinAnalyzer(enable_bmp_analysis=True)
        result = analyzer.analyze_bmp_sequence(bmp_files, club_type)
        
        logger.info(f"BMP 직접 분석 완료: {result.get('total_spin', 0)} rpm")
        return result
        
    except Exception as e:
        logger.error(f"BMP 직접 분석 실패: {e}")
        return {"error": str(e)}

def analyze_bmp_directly_enhanced(source_dir: str, club_type: str = 'driver') -> Dict[str, Any]:
    """
    개선된 가시성으로 BMP 파일을 직접 분석 (딤플 검출)
    
    Args:
        source_dir: BMP 파일이 있는 디렉토리
        club_type: 클럽 타입
        
    Returns:
        분석 결과 딕셔너리
    """
    logger = logging.getLogger(__name__)
    
    try:
        # BMP 로더 생성
        bmp_loader = create_bmp_loader(enable_cache=True)
        
        # BMP 파일 찾기
        bmp_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith('.bmp'):
                    bmp_files.append(os.path.join(root, file))
        
        if not bmp_files:
            logger.warning(f"BMP 파일을 찾을 수 없습니다: {source_dir}")
            return {"error": "BMP 파일을 찾을 수 없습니다"}
        
        # 파일 수 제한 (성능상 이유로)
        if len(bmp_files) > 20:
            bmp_files = bmp_files[:20]
            logger.info(f"처리할 파일 수를 20개로 제한했습니다.")
        
        # 분석기 임포트 및 실행 (개선된 버전)
        from algorithms.spin_analysis.no_dimple_spin_final import FinalNoDimpleSpinAnalyzer
        
        analyzer = FinalNoDimpleSpinAnalyzer(enable_bmp_analysis=True)
        result = analyzer.analyze_bmp_with_improved_visibility(bmp_files, club_type)
        
        logger.info(f"개선된 BMP 직접 분석 완료: {result.get('total_spin', 0)} rpm")
        return result
        
    except Exception as e:
        logger.error(f"개선된 BMP 직접 분석 실패: {e}")
        return {"error": str(e)}

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BMP 처리 도구')
    parser.add_argument('--mode', type=str, choices=['convert', 'analyze'], 
                       default='convert', help='실행 모드')
    parser.add_argument('--source', type=str, default='shot-image', 
                       help='소스 디렉토리')
    parser.add_argument('--target', type=str, default='shot-image-jpg', 
                       help='타겟 디렉토리 (convert 모드)')
    parser.add_argument('--club', type=str, default='driver', 
                       choices=['driver', '7iron'], help='클럽 타입 (analyze 모드)')
    parser.add_argument('--enhanced', action='store_true', 
                       help='개선된 가시성 모드 (analyze 모드)')
    
    args = parser.parse_args()
    
    if args.mode == 'convert':
        # JPG 변환 모드
        converted, skipped, errors = convert_bmp_to_jpg(args.source, args.target)
        
    if errors:
            print(f"변환 완료: {converted}개, 건너뜀: {skipped}개, 오류: {len(errors)}개")
            exit(1)
        else:
            print(f"변환 완료: {converted}개, 건너뜀: {skipped}개")
            exit(0)
    
    elif args.mode == 'analyze':
        # BMP 직접 분석 모드
        if args.enhanced:
            result = analyze_bmp_directly_enhanced(args.source, args.club)
        else:
            result = analyze_bmp_directly(args.source, args.club)
        
        if 'error' in result:
            print(f"분석 실패: {result['error']}")
        exit(1)
    else:
            print(f"BMP 직접 분석 결과:")
            print(f"Total Spin: {result.get('total_spin', 0)} rpm")
            print(f"Backspin: {result.get('backspin', 0)} rpm")
            print(f"Sidespin: {result.get('sidespin', 0)} rpm")
            print(f"신뢰도: {result.get('confidence', 0):.1%}")
        exit(0)

if __name__ == "__main__":
    main()