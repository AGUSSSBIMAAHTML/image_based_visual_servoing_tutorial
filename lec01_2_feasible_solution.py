import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 시뮬레이션 파라미터 설정 ---
# 제어 게인 (λ): 이 값을 키우면 로봇이 더 빠르게 움직임
LAMBDA = 0.5

# 시뮬레이션 시간 설정
DT = 0.1  # 시간 간격
SIM_TIME = 20  # 총 시뮬레이션 시간 (초)
TOTAL_STEPS = int(SIM_TIME / DT)

# 카메라 파라미터 (단위: 픽셀)
FOCAL_LENGTH = 500  # 초점 거리

# --- 목표 설정 ---
# 1. 월드 좌표계에서의 실제 목표 피처 위치 (x, y, z)
# 4개의 점으로 사각형을 정의
target_features_world = np.array([
    [0.5, 0.5, 2.5],
    [0.5, -0.5, 2.5],
    [-0.5, -0.5, 2.5],
    [-0.5, 0.5, 2.5]
])

# 2. 카메라 이미지 평면에서의 '원하는' 피처 위치 (단위: 픽셀)
# 카메라가 목표물 정면, 특정 거리에서 바라볼 때의 이상적인 이미지
desired_features_image = np.array([
    [100, 100],
    [100, -100],
    [-100, -100],
    [-100, 100]
]).flatten() # 1D 벡터로 변환 (8x1)

# --- 로봇(카메라) 초기 상태 ---
# 초기 위치 (x, y) 및 각도 (theta)
initial_pose = np.array([-2.0, 2.0, np.deg2rad(45)])


# --- 헬퍼 함수 ---

def project_to_camera(features_world, camera_pose):
    """월드 좌표계의 3D 점들을 카메라 이미지 평면으로 투영"""
    x, y, theta = camera_pose
    
    # 변환 행렬: 월드 -> 카메라 좌표계
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    t = -R @ np.array([x, y])
    
    projected_points = []
    depths = []
    for p_world in features_world:
        p_world_xy = p_world[:2]
        # 카메라 좌표계로 변환
        p_cam_xy = R @ p_world_xy + t
        Z = p_world[2] # 깊이(Z)는 상수로 가정
        
        # 원근 투영 (Perspective Projection)
        u = FOCAL_LENGTH * p_cam_xy[0] / Z
        v = FOCAL_LENGTH * p_cam_xy[1] / Z
        projected_points.extend([u, v])
        depths.append(Z)
        
    return np.array(projected_points), np.array(depths)

def get_image_jacobian(features_image, depths):
    """이미지 자코비안 (Interaction Matrix) 계산"""
    num_features = len(features_image) // 2
    L = np.zeros((2 * num_features, 3)) # 2N x 3 행렬 (v_x, v_y, w_z 에 대해)

    for i in range(num_features):
        u = features_image[2*i]
        v = features_image[2*i+1]
        Z = depths[i]
        
        # 각 피처에 대한 자코비안 행렬
        L[2*i, :] = [-FOCAL_LENGTH/Z, 0, v]
        L[2*i+1, :] = [0, -FOCAL_LENGTH/Z, -u]
        
    return L

# --- 시뮬레이션 초기화 ---
camera_pose = initial_pose.copy()
pose_history = [camera_pose.copy()]
error_history = []
feature_history = []

# --- 메인 시뮬레이션 루프 ---
for step in range(TOTAL_STEPS):
    # 1. 현재 카메라 위치에서 피처가 어떻게 보이는지 계산 (Sensing)
    current_features, current_depths = project_to_camera(target_features_world, camera_pose)
    feature_history.append(current_features.reshape(-1, 2))

    # 2. 이미지 오차 계산 (Error Calculation) e = s - s*
    error = current_features - desired_features_image
    error_norm = np.linalg.norm(error)
    error_history.append(error_norm)

    # 오차가 충분히 작아지면 시뮬레이션 종료
    if error_norm < 1.0:
        print(f"목표 도달! {step * DT:.2f} 초 소요.")
        break

    # 3. 이미지 자코비안 계산
    L = get_image_jacobian(current_features, current_depths)
    
    # 4. 제어 법칙 적용 (Control Law) v = -λ * L⁺ * e
    # v_c = [v_x, v_y, w_z] 카메라 속도 벡터
    L_inv = np.linalg.pinv(L) # 유사 역행렬(pseudo-inverse) 사용
    camera_velocity = -LAMBDA * L_inv @ error

    # 5. 카메라(로봇) 상태 업데이트
    # 카메라 속도를 월드 좌표계 속도로 변환
    theta = camera_pose[2]
    rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta),  0],
                        [0,             0,              1]])
    
    # 카메라 프레임의 속도를 월드 프레임의 포즈 변화로 변환
    # [vx_world, vy_world, w_world]
    pose_dot = rot_mat @ np.array([camera_velocity[0], camera_velocity[1], 0]) 
    pose_dot[2] = camera_velocity[2] # 각속도는 프레임에 무관

    camera_pose += pose_dot * DT
    pose_history.append(camera_pose.copy())

pose_history = np.array(pose_history)
error_history = np.array(error_history)
feature_history = np.array(feature_history)


# --- 시각화 ---
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 1, 2)

# 서브플롯 설정
ax1.set_title("World View (Top-down)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.grid(True)
ax1.set_aspect('equal')

ax2.set_title("Camera View")
ax2.set_xlabel("u (pixels)")
ax2.set_ylabel("v (pixels)")
ax2.grid(True)
ax2.set_aspect('equal')

ax3.set_title("Image Feature Error Norm")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Error (pixels)")
ax3.grid(True)


# 고정된 요소들 그리기
# World View
ax1.plot(target_features_world[:, 0], target_features_world[:, 1], 'g*', markersize=15, label="Target Features (World)")

# Camera View
des_feat_reshaped = desired_features_image.reshape(-1, 2)
ax2.plot(des_feat_reshaped[:, 0], des_feat_reshaped[:, 1], 'r+', markersize=15, label="Desired Features (s*)")

# 애니메이션을 위한 초기 객체
world_path, = ax1.plot([], [], 'b-', lw=2, label="Robot Path")
robot_body, = ax1.plot([], [], 'ro', markersize=10, label="Robot")
robot_dir, = ax1.plot([], [], 'r-', lw=2)

current_feat_plot, = ax2.plot([], [], 'bo', markersize=10, label="Current Features (s)")
error_plot, = ax3.plot([], [], 'k-', lw=2)

ax1.legend()
ax2.legend()
ax1.axis([-3, 3, -1, 4])
ax2.axis([-250, 250, -250, 250])


def animate(i):
    # World View 업데이트
    world_path.set_data(pose_history[:i+1, 0], pose_history[:i+1, 1])
    robot_body.set_data([pose_history[i, 0]], [pose_history[i, 1]])
    
    # 로봇 방향 표시
    x, y, th = pose_history[i]
    dx = 0.2 * np.cos(th)
    dy = 0.2 * np.sin(th)
    robot_dir.set_data([x, x+dx], [y, y+dy])
    
    # Camera View 업데이트
    if i < len(feature_history):
      current_feat_plot.set_data(feature_history[i, :, 0], feature_history[i, :, 1])

    # Error Plot 업데이트
    times = np.arange(i + 1) * DT
    error_plot.set_data(times, error_history[:i+1])
    ax3.set_xlim(0, SIM_TIME)
    ax3.set_ylim(0, np.max(error_history) * 1.1)
    
    return world_path, robot_body, robot_dir, current_feat_plot, error_plot

ani = FuncAnimation(fig, animate, frames=len(pose_history), interval=DT*1000, blit=True, repeat=False)

# --- 애니메이션 저장 (이 부분을 추가) ---
print("애니메이션을 'ibvs_simulation.mp4' 파일로 저장합니다. 시간이 걸릴 수 있습니다...")
# fps (frames per second)는 영상의 부드러움을 결정합니다. 30 정도가 일반적입니다.
ani.save('ibvs_simulation.mp4', writer='ffmpeg', fps=30, dpi=100)
print("저장이 완료되었습니다.")

plt.tight_layout()
plt.show()
