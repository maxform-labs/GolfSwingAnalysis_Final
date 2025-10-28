import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- 데이터 정의 ---
# '너 말이 정확히 맞아'라는 피드백을 바탕으로,
# 카메라 간의 상대 위치(Y차이 247mm, Z차이 400mm)와
# 바닥으로부터의 절대 높이(Cam2 500mm, Cam1 900mm)가
# 일치한다는 분석을 기반으로 좌표를 설정합니다.
#
# 공과의 빗변 거리(550mm, 950mm)는 이와 모순되므로 시각화에서 제외하고,
# 대신 상대 위치를 명확히 표현합니다.

# 공의 위치 (원점)
ball_pos = np.array([0, 0])

# 카메라 2의 Y위치는 도면에 특정되지 않았으므로, 임의의 값(예: 300)으로 설정합니다.
# 중요한 것은 Z위치와 카메라 1과의 상대 거리입니다.
y_cam2 = 300
z_cam2 = 500
cam2_pos = np.array([y_cam2, z_cam2])

# 카메라 1의 위치는 카메라 2의 상대 위치를 기준으로 계산합니다.
y_cam1 = y_cam2 + 247
z_cam1 = z_cam2 + 400  # (500 + 400 = 900, 도면과 일치)
cam1_pos = np.array([y_cam1, z_cam1])

# 카메라 간의 빗변 거리 (검증용)
# sqrt(247^2 + 400^2) = 470.1mm, 도면의 470mm와 일치합니다.
dist_cams = np.linalg.norm(cam1_pos - cam2_pos)

# --- 1. 2D 평면도 (Y-Z Plane) 생성 ---
plt.figure(figsize=(10, 10))
ax = plt.gca()

# 원점 (공)
plt.plot(ball_pos[0], ball_pos[1], 'ko', markersize=15, label='Golf Ball (Origin)')

# 카메라
plt.plot(cam2_pos[0], cam2_pos[1], 'ks', markersize=12, label='Camera 2')
plt.plot(cam1_pos[0], cam1_pos[1], 'ks', markersize=12, label='Camera 1')

# 공과 카메라를 잇는 선 (빗변)
plt.plot([ball_pos[0], cam1_pos[0]], [ball_pos[1], cam1_pos[1]], 'k-')
plt.plot([ball_pos[0], cam2_pos[0]], [ball_pos[1], cam2_pos[1]], 'k-')
# 참고: 이 선들의 길이는 각각 950, 550이 아닙니다. (모순되는 정보)

# 카메라 간의 관계를 나타내는 보조선 (점선)
# 카메라 1과 2의 상대 위치
plt.plot([cam2_pos[0], cam1_pos[0]], [cam2_pos[1], cam1_pos[1]], 'k--', label=f'Dist: {dist_cams:.1f}mm (≈470mm)')
# 수평/수직 차이
plt.plot([cam2_pos[0], cam1_pos[0]], [cam2_pos[1], cam2_pos[1]], 'k:') # Cam2 높이에서 수평
plt.plot([cam1_pos[0], cam1_pos[0]], [cam2_pos[1], cam1_pos[1]], 'k:') # Cam1 위치에서 수직

# 높이 보조선
plt.plot([cam1_pos[0], 0], [cam1_pos[1], cam1_pos[1]], 'k--') # Cam1 높이선
plt.plot([cam2_pos[0], 0], [cam2_pos[1], cam2_pos[1]], 'k--') # Cam2 높이선

# 치수 레이블
plt.text(cam1_pos[0] + 10, cam1_pos[1] / 2, f'Z = {z_cam1}mm', ha='left')
plt.text(cam2_pos[0] + 10, cam2_pos[1] / 2, f'Z = {z_cam2}mm', ha='left')
plt.text((cam2_pos[0] + cam1_pos[0]) / 2, cam2_pos[1] - 20, f'dY = {y_cam1 - y_cam2}mm', ha='center')
plt.text(cam1_pos[0] + 10, (cam2_pos[1] + cam1_pos[1]) / 2, f'dZ = {z_cam1 - z_cam2}mm', ha='left')

# 축 그리기 (Y축, Z축)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.text(plt.xlim()[1] * 0.9, -30, 'Y 축 (mm)', ha='right')
plt.text(-30, plt.ylim()[1] * 0.9, 'Z 축 (mm)', ha='right', rotation=90)

# X축 방향 표시 (2D 평면에서는 점으로 표시)
plt.plot(0, 0, 'kx', markersize=10, markeredgewidth=2, label='X 축 (타겟 방향, 화면 밖)')

# 플롯 설정
plt.title('Golf Setup (Y-Z Plane Schematic)')
plt.xlabel('Y 축 (mm)')
plt.ylabel('Z 축 (mm)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
# 'equal' 스케일로 설정하여 기하학적 관계가 왜곡되지 않도록 함
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('golf_setup_2d.png')
print("2D Y-Z 평면도 'golf_setup_2d.png' 저장 완료")


# --- 2. 3D 공간 (X-Y-Z) 생성 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D 좌표로 변환 (X=0 평면에 있다고 가정)
ball_pos_3d = np.array([0, 0, 0])
cam1_pos_3d = np.array([0, cam1_pos[0], cam1_pos[1]])
cam2_pos_3d = np.array([0, cam2_pos[0], cam2_pos[1]])

# 공
ax.plot([ball_pos_3d[0]], [ball_pos_3d[1]], [ball_pos_3d[2]], 'ko', markersize=15, label='Golf Ball (Origin)')

# 카메라
ax.plot([cam1_pos_3d[0]], [cam1_pos_3d[1]], [cam1_pos_3d[2]], 'ks', markersize=12, label='Camera 1')
ax.plot([cam2_pos_3d[0]], [cam2_pos_3d[1]], [cam2_pos_3d[2]], 'ks', markersize=12, label='Camera 2')

# 공-카메라 연결선
ax.plot([ball_pos_3d[0], cam1_pos_3d[0]], [ball_pos_3d[1], cam1_pos_3d[1]], [ball_pos_3d[2], cam1_pos_3d[2]], 'k-')
ax.plot([ball_pos_3d[0], cam2_pos_3d[0]], [ball_pos_3d[1], cam2_pos_3d[1]], [ball_pos_3d[2], cam2_pos_3d[2]], 'k-')

# 카메라 간 상대 위치
ax.plot([cam1_pos_3d[0], cam2_pos_3d[0]], [cam1_pos_3d[1], cam2_pos_3d[1]], [cam1_pos_3d[2], cam2_pos_3d[2]], 'k--')

# 카메라 높이 보조선
ax.plot([0, 0], [cam1_pos_3d[1], cam1_pos_3d[1]], [0, cam1_pos_3d[2]], 'k:') # Cam1
ax.plot([0, 0], [cam2_pos_3d[1], cam2_pos_3d[1]], [0, cam2_pos_3d[2]], 'k:') # Cam2

# X축 (타겟 방향)
target_line_length = 1500
ax.plot([0, target_line_length], [0, 0], [0, 0], 'r-', linewidth=2, label='X (Target Direction)')
ax.text(target_line_length, 0, 0, 'X', color='red')

# Y축
ax.plot([0, 0], [0, y_cam1 + 100], [0, 0], 'g-', linewidth=1)
ax.text(0, y_cam1 + 100, 0, 'Y', color='green')

# Z축
ax.plot([0, 0], [0, 0], [0, z_cam1 + 100], 'b-', linewidth=1)
ax.text(0, 0, z_cam1 + 100, 'Z', color='blue')

# 플롯 설정
ax.set_xlabel('X (Target) (mm)')
ax.set_ylabel('Y (Side) (mm)')
ax.set_zlabel('Z (Height) (mm)')
ax.set_title('Golf Setup (3D View)')
ax.legend()
ax.grid(True)

# 3D 뷰의 스케일을 비슷하게 맞춤 (Matplotlib 3D의 한계)
max_range = np.array([target_line_length, cam1_pos_3d[1], cam1_pos_3d[2]]).max()
ax.set_xlim([-max_range*0.1, max_range])
ax.set_ylim([0, max_range])
ax.set_zlim([0, max_range])
# 뷰 각도 조절
ax.view_init(elev=20, azim=120)

plt.tight_layout()
plt.savefig('golf_setup_3d.png')
print("3D 'golf_setup_3d.png' 저장 완료")

# 두 플롯을 함께 보여주기
plt.show()