#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMP Dimple Optimizer v2.0
골프공 딤플 및 클럽 표면 최적화 전처리 시스템
Gamma 렌즈 촬영 이미지 특화 처리
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path

class BMPDimpleOptimizer:
    """BMP 이미지 딤플 및 클럽 표면 최적화 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Gamma 렌즈 특화 파라미터
        self.gamma_lens_params = {
            'gamma_value': 2.2,        # Gamma 렌즈 역보정값
            'exposure_comp': 1.2,      # 노출 보정
            'contrast_boost': 1.8,     # 대비 증폭
            'sharpness': 2.5,         # 선명도 강화
            'denoise_strength': 3,    # 노이즈 제거 강도
            'color_temp': 6500        # 색온도 (K)
        }
        
        # 딤플 검출 최적화 파라미터
        self.dimple_params = {
            'edge_strength': 3.0,      # 엣지 강화 정도
            'local_contrast': 2.5,     # 국소 대비 향상
            'texture_enhance': 1.8,    # 텍스처 강화
            'shadow_comp': 0.7,        # 그림자 보정
            'highlight_comp': 0.8,     # 하이라이트 보정
            'detail_radius': 5,        # 디테일 반경
            'dimple_threshold': 30     # 딤플 검출 임계값
        }
        
        # 클럽 표면 각도 검출 파라미터
        self.club_params = {
            'metallic_enhance': 2.0,   # 금속 표면 강화
            'reflection_reduce': 0.6,  # 반사 감소
            'angle_contrast': 2.5,     # 각도 대비 강화
            'surface_detail': 3.0,     # 표면 디테일
            'groove_enhance': 2.0,     # 홈 강화
            'edge_precision': 4.0      # 엣지 정밀도
        }
        
        # 고급 필터 커널
        self.setup_kernels()
        
    def setup_kernels(self):
        """고급 필터 커널 설정"""
        
        # 딤플 검출용 라플라시안 커널 (강화버전)
        self.dimple_kernel = np.array([
            [-2, -2, -2, -2, -2],
            [-2,  1,  2,  1, -2],
            [-2,  2, 12,  2, -2],
            [-2,  1,  2,  1, -2],
            [-2, -2, -2, -2, -2]
        ], dtype=np.float32) / 16
        
        # 클럽 표면 검출용 방향성 커널
        self.club_kernel_h = np.array([  # 수평 엣지
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)
        
        self.club_kernel_v = np.array([  # 수직 엣지
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
        
        # 텍스처 강화 커널
        self.texture_kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
        
    def process_bmp_for_dimples(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        BMP 이미지 딤플 최적화 처리
        
        Args:
            image_path: 입력 BMP 경로
            output_path: 출력 경로 (선택)
            
        Returns:
            처리된 이미지
        """
        # BMP 직접 읽기 (품질 손실 없음)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
            
        # 원본 백업
        original = image.copy()
        
        # 1. Gamma 렌즈 보정
        image = self.correct_gamma_lens(image)
        
        # 2. 색공간 최적화
        image = self.optimize_color_space(image)
        
        # 3. 딤플 가시성 극대화
        image = self.maximize_dimple_visibility(image)
        
        # 4. 클럽 표면 강화
        image = self.enhance_club_surface(image, original)
        
        # 5. 최종 품질 향상
        image = self.final_quality_enhancement(image)
        
        # 저장
        if output_path:
            cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.logger.info(f"Saved optimized image to {output_path}")
            
        return image
        
    def correct_gamma_lens(self, image: np.ndarray) -> np.ndarray:
        """
        Gamma 렌즈 특성 보정
        
        Args:
            image: 입력 이미지
            
        Returns:
            보정된 이미지
        """
        # 부동소수점 변환
        img_float = image.astype(np.float32) / 255.0
        
        # Gamma 역보정 (렌즈 특성 복원)
        gamma = self.gamma_lens_params['gamma_value']
        img_corrected = np.power(img_float, 1.0 / gamma)
        
        # 노출 보정
        exposure = self.gamma_lens_params['exposure_comp']
        img_corrected = img_corrected * exposure
        
        # 색온도 조정 (Gamma 렌즈 특성)
        if len(image.shape) == 3:
            # BGR 채널 개별 조정
            img_corrected[:,:,0] *= 0.95  # Blue 채널 감소
            img_corrected[:,:,2] *= 1.05  # Red 채널 증가
        
        # 클리핑 및 변환
        img_corrected = np.clip(img_corrected, 0, 1)
        return (img_corrected * 255).astype(np.uint8)
        
    def optimize_color_space(self, image: np.ndarray) -> np.ndarray:
        """
        색공간 최적화 (LAB 색공간 활용)
        
        Args:
            image: 입력 이미지
            
        Returns:
            최적화된 이미지
        """
        if len(image.shape) == 3:
            # LAB 색공간 변환
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # L 채널 (밝기) 최적화
            l_channel = lab[:,:,0]
            
            # CLAHE를 L 채널에만 적용
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l_channel.astype(np.uint8))
            
            # 로컬 대비 향상
            l_enhanced = self.enhance_local_contrast(l_enhanced)
            
            # LAB 재결합
            lab[:,:,0] = l_enhanced
            
            # BGR로 변환
            image = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
        return image
        
    def maximize_dimple_visibility(self, image: np.ndarray) -> np.ndarray:
        """
        딤플 가시성 극대화
        
        Args:
            image: 입력 이미지
            
        Returns:
            딤플이 강조된 이미지
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 1. 멀티스케일 디테일 향상
        details = self.extract_multiscale_details(gray)
        
        # 2. 딤플 특화 필터링
        dimple_filtered = cv2.filter2D(details, -1, self.dimple_kernel)
        
        # 3. 적응형 임계값 처리
        adaptive_thresh = cv2.adaptiveThreshold(
            dimple_filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11, C=2
        )
        
        # 4. 모폴로지 연산으로 딤플 강화
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        dimples = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_GRADIENT, kernel)
        
        # 5. 원본과 블렌딩
        if len(image.shape) == 3:
            # 딤플 마스크를 컬러 이미지에 적용
            dimple_mask = cv2.cvtColor(dimples, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(image, 0.7, dimple_mask, 0.3, 0)
        else:
            result = cv2.addWeighted(gray, 0.7, dimples, 0.3, 0)
            
        return result
        
    def enhance_club_surface(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        """
        클럽 표면 각도 강화
        
        Args:
            image: 현재 처리된 이미지
            original: 원본 이미지
            
        Returns:
            클럽 표면이 강화된 이미지
        """
        # 그레이스케일 작업
        if len(image.shape) == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray = original.copy()
            
        # 1. 방향성 엣지 검출
        edges_h = cv2.filter2D(gray, -1, self.club_kernel_h)
        edges_v = cv2.filter2D(gray, -1, self.club_kernel_v)
        
        # 2. 엣지 강도 계산
        edge_magnitude = np.sqrt(edges_h**2 + edges_v**2)
        edge_angle = np.arctan2(edges_v, edges_h)
        
        # 3. 금속 표면 검출 (높은 반사율 영역)
        metallic_mask = self.detect_metallic_surface(gray)
        
        # 4. 표면 각도 강조
        angle_enhanced = self.enhance_surface_angles(edge_magnitude, edge_angle, metallic_mask)
        
        # 5. 그루브(홈) 검출 및 강화
        grooves = self.enhance_grooves(gray, metallic_mask)
        
        # 6. 결합
        if len(image.shape) == 3:
            # 엣지와 그루브를 컬러 이미지에 적용
            surface_mask = cv2.cvtColor(angle_enhanced + grooves, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(image, 0.8, surface_mask, 0.2, 0)
        else:
            result = cv2.addWeighted(image, 0.8, angle_enhanced + grooves, 0.2, 0)
            
        return result
        
    def extract_multiscale_details(self, gray: np.ndarray) -> np.ndarray:
        """
        멀티스케일 디테일 추출
        
        Args:
            gray: 그레이스케일 이미지
            
        Returns:
            디테일이 추출된 이미지
        """
        # 다양한 스케일의 가우시안 블러
        scales = [1, 2, 4, 8]
        details = np.zeros_like(gray, dtype=np.float32)
        
        for scale in scales:
            # 가우시안 블러
            blurred = cv2.GaussianBlur(gray, (0, 0), scale)
            
            # 디테일 추출 (원본 - 블러)
            detail = gray.astype(np.float32) - blurred.astype(np.float32)
            
            # 가중치 적용
            weight = 1.0 / scale
            details += detail * weight
            
        # 정규화
        details = cv2.normalize(details, None, 0, 255, cv2.NORM_MINMAX)
        return details.astype(np.uint8)
        
    def enhance_local_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        로컬 대비 향상
        
        Args:
            image: 입력 이미지
            
        Returns:
            대비가 향상된 이미지
        """
        # 언샤프 마스킹
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)
        
        # 로컬 히스토그램 균등화
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
        enhanced = clahe.apply(unsharp)
        
        return enhanced
        
    def detect_metallic_surface(self, gray: np.ndarray) -> np.ndarray:
        """
        금속 표면 검출
        
        Args:
            gray: 그레이스케일 이미지
            
        Returns:
            금속 표면 마스크
        """
        # 높은 밝기 영역 검출
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 그라디언트가 낮은 영역 (균일한 표면)
        gradient = cv2.Laplacian(gray, cv2.CV_64F)
        gradient_mag = np.abs(gradient)
        _, smooth = cv2.threshold(gradient_mag, 10, 255, cv2.THRESH_BINARY_INV)
        
        # 결합
        metallic = cv2.bitwise_and(bright, smooth.astype(np.uint8))
        
        # 모폴로지 연산으로 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        metallic = cv2.morphologyEx(metallic, cv2.MORPH_CLOSE, kernel)
        
        return metallic
        
    def enhance_surface_angles(self, magnitude: np.ndarray, angle: np.ndarray, 
                              mask: np.ndarray) -> np.ndarray:
        """
        표면 각도 강조
        
        Args:
            magnitude: 엣지 강도
            angle: 엣지 각도
            mask: 금속 표면 마스크
            
        Returns:
            각도가 강조된 이미지
        """
        # 각도별 강조
        angle_enhanced = np.zeros_like(magnitude)
        
        # 주요 각도 범위 (클럽 페이스 각도)
        angle_ranges = [
            (-np.pi/4, np.pi/4),     # 수평
            (np.pi/4, 3*np.pi/4),     # 대각선
            (-3*np.pi/4, -np.pi/4),   # 역대각선
        ]
        
        for min_angle, max_angle in angle_ranges:
            # 해당 각도 범위의 엣지 강조
            angle_mask = np.logical_and(angle >= min_angle, angle <= max_angle)
            angle_enhanced[angle_mask] = magnitude[angle_mask] * self.club_params['angle_contrast']
            
        # 금속 표면에만 적용
        angle_enhanced = cv2.bitwise_and(angle_enhanced.astype(np.uint8), mask)
        
        return angle_enhanced
        
    def enhance_grooves(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        클럽 페이스 그루브(홈) 강화
        
        Args:
            gray: 그레이스케일 이미지
            mask: 금속 표면 마스크
            
        Returns:
            그루브가 강조된 이미지
        """
        # 수평 라인 검출 (클럽 그루브는 주로 수평)
        horizontal_kernel = np.array([[1,1,1,1,1],
                                     [0,0,0,0,0],
                                     [-1,-1,-1,-1,-1]], dtype=np.float32)
        
        grooves = cv2.filter2D(gray, -1, horizontal_kernel)
        grooves = np.abs(grooves)
        
        # 금속 표면에만 적용
        grooves = cv2.bitwise_and(grooves.astype(np.uint8), mask)
        
        # 대비 강화
        grooves = cv2.convertScaleAbs(grooves, alpha=self.club_params['groove_enhance'], beta=0)
        
        return grooves
        
    def final_quality_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        최종 품질 향상
        
        Args:
            image: 입력 이미지
            
        Returns:
            최종 처리된 이미지
        """
        # 1. 선명도 향상
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # 2. 노이즈 제거 (선택적)
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 
                                                   h=self.gamma_lens_params['denoise_strength'],
                                                   hColor=self.gamma_lens_params['denoise_strength'],
                                                   templateWindowSize=7,
                                                   searchWindowSize=21)
        
        # 3. 최종 대비 조정
        alpha = self.gamma_lens_params['contrast_boost']
        beta = 10  # 밝기 미세 조정
        final = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)
        
        return final
        
    def process_directory(self, input_dir: str, output_dir: str):
        """
        디렉토리 일괄 처리
        
        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # BMP 파일 처리
        for bmp_file in input_path.rglob("*.bmp"):
            relative_path = bmp_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.png')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                self.process_bmp_for_dimples(str(bmp_file), str(output_file))
                self.logger.info(f"Processed: {bmp_file} -> {output_file}")
            except Exception as e:
                self.logger.error(f"Error processing {bmp_file}: {e}")


def main():
    """테스트 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BMP Dimple Optimizer v2.0")
    parser.add_argument("--input", "-i", required=True, help="Input BMP directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 처리 실행
    optimizer = BMPDimpleOptimizer()
    optimizer.process_directory(args.input, args.output)
    print(f"Processing complete. Output saved to {args.output}")


if __name__ == "__main__":
    main()