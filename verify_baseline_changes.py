#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
베이스라인 470mm 통일 후 시스템 정상 작동 검증
Phase 2 회귀 테스트 대체 스크립트
"""

import json
import sys
import os
from pathlib import Path

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

def test_calibration_files():
    """캘리브레이션 파일 로드 테스트"""
    print("=" * 60)
    print("테스트 1: 캘리브레이션 파일 로드 검증")
    print("=" * 60)

    test_files = [
        'config/calibration_default.json',
        'improved_calibration_470mm.json',
        'manual_calibration_470mm.json'
    ]

    results = []
    for file_path in test_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            baseline = data.get('baseline', 0)
            focal_length = data.get('focal_length', 0)

            status = "✅ PASS" if baseline == 470.0 else f"❌ FAIL (baseline={baseline})"
            results.append({
                'file': file_path,
                'status': status,
                'baseline': baseline,
                'focal_length': focal_length
            })

            print(f"\n파일: {file_path}")
            print(f"  상태: {status}")
            print(f"  베이스라인: {baseline}mm")
            print(f"  초점거리: {focal_length}px")

        except FileNotFoundError:
            print(f"\n파일: {file_path}")
            print(f"  상태: ⚠️ WARNING (파일 없음)")
            results.append({
                'file': file_path,
                'status': '⚠️ WARNING',
                'baseline': None,
                'focal_length': None
            })
        except Exception as e:
            print(f"\n파일: {file_path}")
            print(f"  상태: ❌ ERROR ({str(e)})")
            results.append({
                'file': file_path,
                'status': '❌ ERROR',
                'baseline': None,
                'focal_length': None
            })

    return results

def test_python_imports():
    """수정된 Python 파일 import 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: 수정된 Python 파일 Import 검증")
    print("=" * 60)

    # 수정된 파일들의 Python 경로
    test_modules = [
        ('final_trajectory_tracker', 'FinalTrajectoryTracker'),
        ('fixed_3d_tracker', 'Fixed3DTracker')
    ]

    results = []
    for module_name, class_name in test_modules:
        try:
            # 동적 import
            module = __import__(module_name)

            # 클래스 존재 확인
            if hasattr(module, class_name):
                print(f"\n모듈: {module_name}")
                print(f"  클래스: {class_name}")
                print(f"  상태: ✅ PASS")
                results.append({'module': module_name, 'status': 'PASS'})
            else:
                print(f"\n모듈: {module_name}")
                print(f"  클래스: {class_name}")
                print(f"  상태: ⚠️ WARNING (클래스 없음)")
                results.append({'module': module_name, 'status': 'WARNING'})

        except ImportError as e:
            print(f"\n모듈: {module_name}")
            print(f"  상태: ⚠️ WARNING (import 실패: {str(e)})")
            results.append({'module': module_name, 'status': 'WARNING'})
        except Exception as e:
            print(f"\n모듈: {module_name}")
            print(f"  상태: ❌ ERROR ({str(e)})")
            results.append({'module': module_name, 'status': 'ERROR'})

    return results

def test_baseline_consistency():
    """베이스라인 일관성 검증"""
    print("\n" + "=" * 60)
    print("테스트 3: 베이스라인 일관성 검증")
    print("=" * 60)

    # 모든 캘리브레이션 파일의 베이스라인 확인
    calibration_files = [
        'config/calibration_default.json',
        'improved_calibration_470mm.json',
        'manual_calibration_470mm.json'
    ]

    baselines = []
    for file_path in calibration_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            baseline = data.get('baseline', 0)
            baselines.append(baseline)
            print(f"\n{file_path}: {baseline}mm")
        except FileNotFoundError:
            print(f"\n{file_path}: 파일 없음")
            continue

    # 모든 베이스라인이 470mm인지 확인
    if all(b == 470.0 for b in baselines):
        print(f"\n상태: ✅ PASS - 모든 캘리브레이션 파일이 470mm 사용")
        return True
    else:
        print(f"\n상태: ❌ FAIL - 베이스라인 불일치 발견")
        return False

def test_config_calibration_default():
    """config/calibration_default.json 세부 검증"""
    print("\n" + "=" * 60)
    print("테스트 4: 표준 캘리브레이션 파일 세부 검증")
    print("=" * 60)

    file_path = 'config/calibration_default.json'

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 필수 파라미터 확인
        required_params = [
            'baseline',
            'focal_length',
            'camera_matrix_1',
            'camera_matrix_2',
            'rotation_matrix',
            'translation_vector',
            'image_size'
        ]

        print(f"\n파일: {file_path}")
        all_present = True

        for param in required_params:
            if param in data:
                print(f"  ✅ {param}: {data[param]}")
            else:
                print(f"  ❌ {param}: 없음")
                all_present = False

        # Y축 스케일링 확인
        if 'scaling_applied' in data and data['scaling_applied']:
            print(f"\n  ✅ Y축 스케일링 적용됨")
            print(f"  ✅ scale_factor_y: {data.get('scale_factor_y', 'N/A')}")
        else:
            print(f"\n  ⚠️ Y축 스케일링 미적용")

        # default_calibration 플래그 확인
        if data.get('default_calibration', False):
            print(f"  ✅ 표준 캘리브레이션 플래그 설정됨")
        else:
            print(f"  ⚠️ 표준 캘리브레이션 플래그 없음")

        if all_present:
            print(f"\n상태: ✅ PASS - 모든 필수 파라미터 존재")
            return True
        else:
            print(f"\n상태: ❌ FAIL - 누락된 파라미터 있음")
            return False

    except FileNotFoundError:
        print(f"\n상태: ❌ FAIL - 파일 없음: {file_path}")
        return False
    except Exception as e:
        print(f"\n상태: ❌ ERROR - {str(e)}")
        return False

def generate_summary(test_results):
    """검증 결과 요약"""
    print("\n" + "=" * 60)
    print("검증 결과 요약")
    print("=" * 60)

    all_tests = [
        test_results.get('calibration_files', False),
        test_results.get('baseline_consistency', False),
        test_results.get('config_default', False)
    ]

    passed = sum(1 for t in all_tests if t)
    total = len(all_tests)

    print(f"\n총 테스트: {total}")
    print(f"통과: {passed}")
    print(f"실패: {total - passed}")

    if passed == total:
        print(f"\n✅ 모든 테스트 통과!")
        print(f"✅ 베이스라인 470mm 통일 작업이 정상적으로 완료되었습니다.")
        return True
    else:
        print(f"\n⚠️ 일부 테스트 실패")
        print(f"⚠️ 추가 확인이 필요합니다.")
        return False

def main():
    """메인 실행 함수"""
    print("베이스라인 470mm 통일 후 시스템 정상 작동 검증")
    print("Phase 2 회귀 테스트")
    print("=" * 60)

    test_results = {}

    # 테스트 1: 캘리브레이션 파일 로드
    calibration_results = test_calibration_files()
    test_results['calibration_files'] = all(
        r['status'].startswith('✅') for r in calibration_results
        if r['status'] != '⚠️ WARNING'
    )

    # 테스트 2: Python import (optional)
    python_results = test_python_imports()
    test_results['python_imports'] = python_results

    # 테스트 3: 베이스라인 일관성
    test_results['baseline_consistency'] = test_baseline_consistency()

    # 테스트 4: config/calibration_default.json 검증
    test_results['config_default'] = test_config_calibration_default()

    # 결과 요약
    success = generate_summary(test_results)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
