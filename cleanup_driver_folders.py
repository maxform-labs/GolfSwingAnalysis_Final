#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
드라이버 촬영 이미지 정리 스크립트
Gamma_로 시작하는 파일만 남기고 나머지 삭제
"""

import os
import sys
from pathlib import Path

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def analyze_folders(base_path, start_folder=1, end_folder=20):
    """1-20번 폴더 분석"""
    print("=" * 80)
    print("드라이버 이미지 폴더 분석 (1-20번)")
    print("=" * 80)

    total_gamma_files = 0
    total_other_files = 0
    total_size_to_delete = 0

    folder_info = []

    for i in range(start_folder, end_folder + 1):
        folder_path = Path(base_path) / str(i)

        if not folder_path.exists():
            print(f"\n⚠️ 폴더 {i}: 존재하지 않음")
            continue

        gamma_files = []
        other_files = []

        for file in folder_path.iterdir():
            if file.is_file():
                if file.name.startswith('Gamma_'):
                    gamma_files.append(file)
                else:
                    other_files.append(file)

        gamma_count = len(gamma_files)
        other_count = len(other_files)
        other_size = sum(f.stat().st_size for f in other_files)

        total_gamma_files += gamma_count
        total_other_files += other_count
        total_size_to_delete += other_size

        folder_info.append({
            'folder': i,
            'gamma_count': gamma_count,
            'other_count': other_count,
            'other_size': other_size,
            'other_files': other_files
        })

        print(f"\n폴더 {i}:")
        print(f"  ✅ Gamma_ 파일: {gamma_count}개 (유지)")
        print(f"  ❌ 기타 파일: {other_count}개 (삭제 예정)")
        if other_count > 0:
            print(f"  📦 삭제 용량: {other_size / 1024 / 1024:.1f} MB")
            # 샘플 파일 이름 표시 (최대 5개)
            sample_files = [f.name for f in other_files[:5]]
            print(f"  📄 샘플: {', '.join(sample_files)}")
            if other_count > 5:
                print(f"       ... 외 {other_count - 5}개")

    print("\n" + "=" * 80)
    print("전체 요약")
    print("=" * 80)
    print(f"총 Gamma_ 파일: {total_gamma_files}개 (유지)")
    print(f"총 기타 파일: {total_other_files}개 (삭제 예정)")
    print(f"삭제 예정 용량: {total_size_to_delete / 1024 / 1024:.1f} MB")

    return folder_info

def delete_non_gamma_files(folder_info, dry_run=True):
    """Gamma_ 파일이 아닌 파일 삭제"""
    print("\n" + "=" * 80)
    if dry_run:
        print("시뮬레이션 모드 (실제 삭제 안 함)")
    else:
        print("⚠️ 실제 삭제 시작!")
    print("=" * 80)

    deleted_count = 0
    deleted_size = 0

    for info in folder_info:
        folder_num = info['folder']
        other_files = info['other_files']

        if len(other_files) == 0:
            continue

        print(f"\n폴더 {folder_num}: {len(other_files)}개 파일 삭제 중...")

        for file in other_files:
            try:
                file_size = file.stat().st_size

                if not dry_run:
                    file.unlink()  # 실제 삭제
                    print(f"  ✅ 삭제: {file.name}")
                else:
                    print(f"  [시뮬레이션] 삭제 예정: {file.name}")

                deleted_count += 1
                deleted_size += file_size

            except Exception as e:
                print(f"  ❌ 오류: {file.name} - {str(e)}")

    print("\n" + "=" * 80)
    print("삭제 완료 요약")
    print("=" * 80)
    print(f"삭제된 파일: {deleted_count}개")
    print(f"확보된 용량: {deleted_size / 1024 / 1024:.1f} MB")

    return deleted_count, deleted_size

def verify_cleanup(base_path, start_folder=1, end_folder=20):
    """정리 후 검증"""
    print("\n" + "=" * 80)
    print("정리 검증")
    print("=" * 80)

    all_clean = True

    for i in range(start_folder, end_folder + 1):
        folder_path = Path(base_path) / str(i)

        if not folder_path.exists():
            continue

        non_gamma_files = []
        for file in folder_path.iterdir():
            if file.is_file() and not file.name.startswith('Gamma_'):
                non_gamma_files.append(file.name)

        if len(non_gamma_files) > 0:
            print(f"\n⚠️ 폴더 {i}: Gamma_가 아닌 파일이 {len(non_gamma_files)}개 남아있음")
            for fname in non_gamma_files[:5]:
                print(f"    - {fname}")
            all_clean = False
        else:
            print(f"✅ 폴더 {i}: 정리 완료 (Gamma_ 파일만 존재)")

    print("\n" + "=" * 80)
    if all_clean:
        print("✅ 모든 폴더 정리 완료!")
    else:
        print("⚠️ 일부 폴더에 Gamma_가 아닌 파일이 남아있습니다.")
    print("=" * 80)

    return all_clean

def main():
    """메인 실행 함수"""
    base_path = r"C:\src\GolfSwingAnalysis_Final\data\1440_300_data\driver"

    print("드라이버 이미지 폴더 정리 스크립트")
    print("=" * 80)
    print(f"대상 경로: {base_path}")
    print(f"대상 폴더: 1-20번")
    print(f"작업 내용: Gamma_ 파일만 남기고 나머지 삭제")
    print("=" * 80)

    # 1단계: 분석
    print("\n[1단계] 폴더 분석 중...")
    folder_info = analyze_folders(base_path, start_folder=1, end_folder=20)

    # 2단계: 시뮬레이션 (dry run)
    print("\n[2단계] 시뮬레이션 실행 중...")
    delete_non_gamma_files(folder_info, dry_run=True)

    # 3단계: 사용자 확인
    print("\n" + "=" * 80)
    print("위 파일들을 삭제하시겠습니까?")
    print("⚠️ 이 작업은 되돌릴 수 없습니다!")
    print("=" * 80)

    # 자동 실행 모드인 경우
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        print("--execute 플래그 감지: 자동 실행 모드")
        user_input = 'y'
    else:
        user_input = input("계속하시겠습니까? (y/N): ").strip().lower()

    if user_input == 'y':
        # 4단계: 실제 삭제
        print("\n[3단계] 실제 삭제 실행 중...")
        deleted_count, deleted_size = delete_non_gamma_files(folder_info, dry_run=False)

        # 5단계: 검증
        print("\n[4단계] 정리 검증 중...")
        verify_cleanup(base_path, start_folder=1, end_folder=20)

        print("\n✅ 작업 완료!")
        print(f"✅ 총 {deleted_count}개 파일 삭제")
        print(f"✅ 총 {deleted_size / 1024 / 1024:.1f} MB 용량 확보")
    else:
        print("\n작업 취소됨")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
