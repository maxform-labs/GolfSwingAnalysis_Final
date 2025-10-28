import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
from pathlib import Path

class VerticalStereoCalibration:
    def __init__(self, calibration_images_path):
        """
        수직 스테레오 비전 캘리브레이션 클래스
        
        Args:
            calibration_images_path: 캘리브레이션 이미지 폴더 경로
        """
        self.calibration_images_path = calibration_images_path
        self.pattern_size = (9, 6)  # 체스보드 패턴 크기
        self.square_size = 30.0  # 체스보드 사각형 크기 (mm)
        
        # 캘리브레이션 결과 저장
        self.calibration_results = {}
        
    def load_calibration_images(self):
        """캘리브레이션 이미지 로드"""
        print("캘리브레이션 이미지 로딩 중...")
        
        # Cam1과 Cam2 이미지 쌍 찾기
        cam1_images = []
        cam2_images = []
        
        for filename in sorted(os.listdir(self.calibration_images_path)):
            if filename.startswith('Cam1_') and filename.endswith('.bmp'):
                cam1_path = os.path.join(self.calibration_images_path, filename)
                cam2_filename = filename.replace('Cam1_', 'Cam2_')
                cam2_path = os.path.join(self.calibration_images_path, cam2_filename)
                
                if os.path.exists(cam2_path):
                    cam1_images.append(cam1_path)
                    cam2_images.append(cam2_path)
        
        print(f"총 {len(cam1_images)}개의 이미지 쌍을 찾았습니다.")
        return cam1_images, cam2_images
    
    def detect_chessboard_corners(self, image_path):
        """체스보드 코너 검출"""
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            # 서브픽셀 정확도로 코너 위치 개선
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return ret, corners
    
    def create_object_points(self):
        """3D 객체 포인트 생성"""
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def calibrate_individual_cameras(self, cam1_images, cam2_images):
        """개별 카메라 캘리브레이션"""
        print("개별 카메라 캘리브레이션 수행 중...")
        
        objp = self.create_object_points()
        obj_points = []
        img_points_cam1 = []
        img_points_cam2 = []
        
        valid_pairs = 0
        
        for i, (cam1_path, cam2_path) in enumerate(zip(cam1_images, cam2_images)):
            print(f"처리 중: {i+1}/{len(cam1_images)}")
            
            # Cam1 체스보드 검출
            ret1, corners1 = self.detect_chessboard_corners(cam1_path)
            ret2, corners2 = self.detect_chessboard_corners(cam2_path)
            
            if ret1 and ret2:
                obj_points.append(objp)
                img_points_cam1.append(corners1)
                img_points_cam2.append(corners2)
                valid_pairs += 1
                
                # 시각화
                img1 = cv2.imread(cam1_path)
                img2 = cv2.imread(cam2_path)
                cv2.drawChessboardCorners(img1, self.pattern_size, corners1, ret1)
                cv2.drawChessboardCorners(img2, self.pattern_size, corners2, ret2)
                
                # 결과 저장
                cv2.imwrite(f'calibration_debug_cam1_{i+1}.jpg', img1)
                cv2.imwrite(f'calibration_debug_cam2_{i+1}.jpg', img2)
        
        print(f"유효한 이미지 쌍: {valid_pairs}개")
        
        if valid_pairs < 10:
            raise ValueError(f"캘리브레이션에 충분한 이미지가 없습니다. (최소 10개 필요, 현재 {valid_pairs}개)")
        
        # Cam1 캘리브레이션
        print("Cam1 캘리브레이션 중...")
        ret1, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(
            obj_points, img_points_cam1, (1440, 300), None, None
        )
        
        # Cam2 캘리브레이션
        print("Cam2 캘리브레이션 중...")
        ret2, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(
            obj_points, img_points_cam2, (1440, 300), None, None
        )
        
        return {
            'K1': K1, 'D1': D1, 'rvecs1': rvecs1, 'tvecs1': tvecs1,
            'K2': K2, 'D2': D2, 'rvecs2': rvecs2, 'tvecs2': tvecs2,
            'obj_points': obj_points,
            'img_points_cam1': img_points_cam1,
            'img_points_cam2': img_points_cam2
        }
    
    def calibrate_stereo(self, individual_results):
        """스테레오 캘리브레이션"""
        print("스테레오 캘리브레이션 수행 중...")
        
        # 스테레오 캘리브레이션
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            individual_results['obj_points'],
            individual_results['img_points_cam1'],
            individual_results['img_points_cam2'],
            individual_results['K1'],
            individual_results['D1'],
            individual_results['K2'],
            individual_results['D2'],
            (1440, 300),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        # 스테레오 정규화
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, (1440, 300), R, T
        )
        
        return {
            'K1': K1, 'D1': D1, 'K2': K2, 'D2': D2,
            'R': R, 'T': T, 'E': E, 'F': F,
            'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
            'roi1': roi1, 'roi2': roi2,
            'baseline': np.linalg.norm(T),
            'reprojection_error': ret
        }
    
    def run_calibration(self):
        """전체 캘리브레이션 실행"""
        print("=== 수직 스테레오 비전 캘리브레이션 시작 ===")
        
        # 1. 이미지 로드
        cam1_images, cam2_images = self.load_calibration_images()
        
        # 2. 개별 카메라 캘리브레이션
        individual_results = self.calibrate_individual_cameras(cam1_images, cam2_images)
        
        # 3. 스테레오 캘리브레이션
        stereo_results = self.calibrate_stereo(individual_results)
        
        # 4. 결과 저장
        self.calibration_results = stereo_results
        
        # 5. 결과 출력
        self.print_calibration_results()
        
        return stereo_results
    
    def print_calibration_results(self):
        """캘리브레이션 결과 출력"""
        print("\n=== 캘리브레이션 결과 ===")
        print(f"재투영 오차: {self.calibration_results['reprojection_error']:.4f} 픽셀")
        print(f"베이스라인: {self.calibration_results['baseline']:.2f} mm")
        
        print("\nCam1 내부 파라미터:")
        print(f"초점거리: fx={self.calibration_results['K1'][0,0]:.2f}, fy={self.calibration_results['K1'][1,1]:.2f}")
        print(f"주점: cx={self.calibration_results['K1'][0,2]:.2f}, cy={self.calibration_results['K1'][1,2]:.2f}")
        
        print("\nCam2 내부 파라미터:")
        print(f"초점거리: fx={self.calibration_results['K2'][0,0]:.2f}, fy={self.calibration_results['K2'][1,1]:.2f}")
        print(f"주점: cx={self.calibration_results['K2'][0,2]:.2f}, cy={self.calibration_results['K2'][1,2]:.2f}")
        
        print("\n스테레오 변환:")
        print(f"회전 행렬 R:\n{self.calibration_results['R']}")
        print(f"변위 벡터 T: {self.calibration_results['T'].flatten()}")
    
    def save_calibration_results(self, filename='stereo_calibration.json'):
        """캘리브레이션 결과 저장"""
        # NumPy 배열을 리스트로 변환
        results_to_save = {}
        for key, value in self.calibration_results.items():
            if isinstance(value, np.ndarray):
                results_to_save[key] = value.tolist()
            else:
                results_to_save[key] = value
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"캘리브레이션 결과가 {filename}에 저장되었습니다.")
    
    def visualize_calibration_setup(self):
        """캘리브레이션 설정 시각화"""
        # 설계서의 좌표계에 따른 시각화
        fig = plt.figure(figsize=(15, 10))
        
        # 2D 평면도 (Y-Z Plane)
        ax1 = plt.subplot(1, 2, 1)
        
        # 공의 위치 (원점)
        ball_pos = np.array([0, 0])
        plt.plot(ball_pos[0], ball_pos[1], 'ko', markersize=15, label='Golf Ball (Origin)')
        
        # 카메라 위치 (설계서 기준)
        y_cam2 = 300
        z_cam2 = 500
        cam2_pos = np.array([y_cam2, z_cam2])
        
        y_cam1 = y_cam2 + 247
        z_cam1 = z_cam2 + 400
        cam1_pos = np.array([y_cam1, z_cam1])
        
        plt.plot(cam2_pos[0], cam2_pos[1], 'bs', markersize=12, label='Camera 2 (Bottom)')
        plt.plot(cam1_pos[0], cam1_pos[1], 'rs', markersize=12, label='Camera 1 (Top)')
        
        # 연결선
        plt.plot([ball_pos[0], cam1_pos[0]], [ball_pos[1], cam1_pos[1]], 'r-', alpha=0.7)
        plt.plot([ball_pos[0], cam2_pos[0]], [ball_pos[1], cam2_pos[1]], 'b-', alpha=0.7)
        plt.plot([cam2_pos[0], cam1_pos[0]], [cam2_pos[1], cam1_pos[1]], 'k--', alpha=0.5)
        
        # 치수 표시
        plt.text(cam1_pos[0] + 10, cam1_pos[1] / 2, f'Z = {z_cam1}mm', ha='left')
        plt.text(cam2_pos[0] + 10, cam2_pos[1] / 2, f'Z = {z_cam2}mm', ha='left')
        plt.text((cam2_pos[0] + cam1_pos[0]) / 2, cam2_pos[1] - 20, f'dY = {y_cam1 - y_cam2}mm', ha='center')
        plt.text(cam1_pos[0] + 10, (cam2_pos[1] + cam1_pos[1]) / 2, f'dZ = {z_cam1 - z_cam2}mm', ha='left')
        
        plt.title('Vertical Stereo Setup (Y-Z Plane)')
        plt.xlabel('Y (mm)')
        plt.ylabel('Z (mm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 3D 시각화
        ax2 = plt.subplot(1, 2, 2, projection='3d')
        
        # 3D 좌표
        ball_3d = np.array([0, 0, 0])
        cam1_3d = np.array([0, y_cam1, z_cam1])
        cam2_3d = np.array([0, y_cam2, z_cam2])
        
        ax2.plot([ball_3d[0]], [ball_3d[1]], [ball_3d[2]], 'ko', markersize=15, label='Golf Ball')
        ax2.plot([cam1_3d[0]], [cam1_3d[1]], [cam1_3d[2]], 'rs', markersize=12, label='Cam1 (Top)')
        ax2.plot([cam2_3d[0]], [cam2_3d[1]], [cam2_3d[2]], 'bs', markersize=12, label='Cam2 (Bottom)')
        
        # 연결선
        ax2.plot([ball_3d[0], cam1_3d[0]], [ball_3d[1], cam1_3d[1]], [ball_3d[2], cam1_3d[2]], 'r-', alpha=0.7)
        ax2.plot([ball_3d[0], cam2_3d[0]], [ball_3d[1], cam2_3d[1]], [ball_3d[2], cam2_3d[2]], 'b-', alpha=0.7)
        ax2.plot([cam1_3d[0], cam2_3d[0]], [cam1_3d[1], cam2_3d[1]], [cam1_3d[2], cam2_3d[2]], 'k--', alpha=0.5)
        
        # 축 표시
        ax2.plot([0, 1000], [0, 0], [0, 0], 'r-', linewidth=2, label='X (Target)')
        ax2.plot([0, 0], [0, 600], [0, 0], 'g-', linewidth=1, label='Y')
        ax2.plot([0, 0], [0, 0], [0, 1000], 'b-', linewidth=1, label='Z')
        
        ax2.set_xlabel('X (Target)')
        ax2.set_ylabel('Y (Side)')
        ax2.set_zlabel('Z (Height)')
        ax2.set_title('3D Stereo Setup')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('calibration_setup_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("캘리브레이션 설정 시각화가 'calibration_setup_visualization.png'에 저장되었습니다.")


def main():
    """메인 실행 함수"""
    # 캘리브레이션 이미지 경로
    calibration_path = "data2/calibrations"
    
    # 캘리브레이션 실행
    calibrator = VerticalStereoCalibration(calibration_path)
    
    try:
        # 캘리브레이션 수행
        results = calibrator.run_calibration()
        
        # 결과 저장
        calibrator.save_calibration_results()
        
        # 설정 시각화
        calibrator.visualize_calibration_setup()
        
        print("\n=== 캘리브레이션 완료 ===")
        print("다음 단계: 캘리브레이션 검증을 수행하세요.")
        
    except Exception as e:
        print(f"캘리브레이션 중 오류 발생: {e}")
        return None
    
    return results


if __name__ == "__main__":
    results = main()

