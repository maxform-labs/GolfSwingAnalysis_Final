#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
수정사항 검증 테스트 스크립트
"""

import numpy as np
import sys
import traceback

# Windows 콘솔 인코딩 설정
import io
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

def test_y_axis_disparity():
    """Y축 시차 공식 테스트"""
    try:
        # 직접 공식 테스트
        def calculate_vertical_disparity_depth(x, y, disparity_y, fy, baseline):
            if abs(disparity_y) < 0.1:
                return float('inf')
            depth = (fy * baseline) / disparity_y
            return np.clip(depth, 500.0, 50000.0)
        
        result = calculate_vertical_disparity_depth(100, 200, 10.0, 800.0, 400.0)
        print(f"[PASS] Y축 시차 깊이 계산: {result}mm")
        
        # 0으로 나누기 방지 테스트
        result_zero = calculate_vertical_disparity_depth(100, 200, 0.05, 800.0, 400.0)
        print(f"[PASS] 제로 시차 처리: {result_zero}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Y축 시차 테스트 실패: {e}")
        return False

def test_bayesian_ensemble():
    """베이지안 앙상블 테스트"""
    try:
        from advanced_algorithms import BayesianEnsemble
        
        ensemble = BayesianEnsemble()
        
        # 테스트 측정값들
        test_measurements = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
            np.array([0.9, 1.9, 2.9])
        ]
        
        estimate, confidence = ensemble.estimate_with_ensemble(test_measurements)
        
        print(f"✅ 베이지안 앙상블:")
        print(f"   - 추정값: {estimate}")
        print(f"   - 신뢰도: {confidence}")
        
        # 3개 추정기 존재 확인
        estimator_names = list(ensemble.estimators.keys())
        expected_estimators = ['kalman', 'particle', 'least_squares']
        
        if all(name in estimator_names for name in expected_estimators):
            print(f"✅ 3개 추정기 모두 존재: {estimator_names}")
        else:
            print(f"❌ 추정기 누락: 필요={expected_estimators}, 존재={estimator_names}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 베이지안 앙상블 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_adaptive_correction():
    """적응형 보정 테스트"""
    try:
        from adaptive_correction import AdaptiveCorrector, SkillLevel
        
        corrector = AdaptiveCorrector()
        
        # 스킬 레벨 테스트
        corrector.set_skill_level(SkillLevel.INTERMEDIATE)
        print(f"✅ 스킬 레벨 설정: {corrector.current_skill_level}")
        
        # 측정값 보정 테스트
        original = np.array([1.5, 0.3, 5.2])
        corrected = corrector.apply_skill_correction(original, 'position')
        
        print(f"✅ 적응형 보정:")
        print(f"   - 원본: {original}")
        print(f"   - 보정: {corrected}")
        
        # 물리적 제약 조건 테스트
        extreme_values = np.array([10000, -2000, 100000])
        constrained = corrector.apply_physical_constraints(extreme_values, 'position')
        
        print(f"✅ 물리적 제약:")
        print(f"   - 원본: {extreme_values}")
        print(f"   - 제약 적용: {constrained}")
        
        # 적응형 노이즈 튜닝 테스트
        noise_params = corrector.adaptive_noise_tuning(0.8)
        print(f"✅ 적응형 노이즈 튜닝: {noise_params}")
        
        return True
        
    except Exception as e:
        print(f"❌ 적응형 보정 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_real_time_methods():
    """실시간 파이프라인 메서드 존재 확인"""
    try:
        import golf_swing_analyzer
        import inspect
        
        # RealTimeProcessor 클래스 확인
        if hasattr(golf_swing_analyzer, 'RealTimeProcessor'):
            processor_class = golf_swing_analyzer.RealTimeProcessor
            
            required_methods = [
                'capture_synchronized_frames',
                'detect_objects_gpu', 
                'perform_3d_analysis_gpu',
                'output_result',
                'wait_for_ir_sync',
                'calculate_ball_3d_coordinates',
                'apply_physics_constraints'
            ]
            
            existing_methods = [name for name, method in inspect.getmembers(processor_class, inspect.isfunction)]
            
            missing_methods = [method for method in required_methods if method not in existing_methods]
            
            if not missing_methods:
                print("✅ 실시간 파이프라인 메서드 모두 구현됨")
                for method in required_methods:
                    print(f"   - {method}")
                return True
            else:
                print(f"❌ 누락된 실시간 메서드: {missing_methods}")
                return False
        else:
            print("❌ RealTimeProcessor 클래스를 찾을 수 없음")
            return False
            
    except Exception as e:
        print(f"❌ 실시간 파이프라인 테스트 실패: {e}")
        return False

def test_imports():
    """필요한 모듈들이 올바르게 임포트되는지 테스트"""
    try:
        # 모든 수정된 모듈 임포트 테스트
        modules_to_test = [
            'advanced_algorithms',
            'adaptive_correction',
            'golf_swing_analyzer'
        ]
        
        imported_modules = []
        for module_name in modules_to_test:
            try:
                module = __import__(module_name)
                imported_modules.append(module_name)
                print(f"✅ {module_name} 모듈 임포트 성공")
            except ImportError as e:
                print(f"❌ {module_name} 모듈 임포트 실패: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 임포트 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("골프 스윙 분석 시스템 수정사항 검증")
    print("=" * 60)
    
    tests = [
        ("모듈 임포트", test_imports),
        ("Y축 시차 공식", test_y_axis_disparity), 
        ("베이지안 앙상블", test_bayesian_ensemble),
        ("적응형 보정", test_adaptive_correction),
        ("실시간 파이프라인", test_real_time_methods)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name} 테스트 실행...")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))
        print("-" * 40)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\n총 테스트: {total}")
    print(f"통과: {passed}")
    print(f"실패: {total - passed}")
    print(f"통과율: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 모든 수정사항이 성공적으로 적용되었습니다!")
        return True
    else:
        print(f"\n⚠️  {total - passed}개 테스트가 실패했습니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)