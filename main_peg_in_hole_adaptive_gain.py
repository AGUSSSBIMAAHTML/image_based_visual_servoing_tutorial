import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Params ---
DT = 0.1
SIM_TIME = 20
TOTAL_STEPS = int(SIM_TIME / DT)
FOCAL_LENGTH = 500.0
LAMBDA_MAX, LAMBDA_MIN = 2.0, 0.5
K_YAW = 0.5   # yaw proportional gain

# --- Helpers (rotations) ---
def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def Rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

# --- Geometry ---
def make_rim_world(center_w, radius=0.5, tilt_rx_deg=20.0, tilt_ry_deg=5.0, n=200):
    rx = np.deg2rad(tilt_rx_deg); ry = np.deg2rad(tilt_ry_deg)
    c,s = np.cos, np.sin
    Rx = np.array([[1,0,0],[0,c(rx),-s(rx)],[0,s(rx),c(rx)]])
    Ry = np.array([[c(ry),0,s(ry)],[0,1,0],[-s(ry),0,c(ry)]])
    R_plane = Ry @ Rx
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring_local = np.stack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)], axis=1)
    return (R_plane @ ring_local.T).T + center_w

def project_points_to_image(pts_w, pose_xyth, fpx):
    x, y, th = pose_xyth
    Rcw = Rz(th).T
    tcw = -Rcw @ np.array([x, y, 0.0])
    pts_c = (Rcw @ pts_w.T).T + tcw
    X, Y, Z = pts_c[:,0], pts_c[:,1], pts_c[:,2]
    return np.stack([fpx*(X/Z), fpx*(Y/Z)], axis=1), Z

# --- Features (circle only) ---
def circle_from_points(uv):
    u0, v0 = uv.mean(axis=0)
    duv = uv - np.array([u0, v0])
    r = np.sqrt(np.mean(np.sum(duv**2, axis=1)))
    return float(u0), float(v0), float(r)

def features_from_circle(u0, v0, r):
    return np.array([u0, v0, np.log(max(r*r, 1e-12))])

def get_features(pose_xyth, rim_w, fpx):
    uv, _ = project_points_to_image(rim_w, pose_xyth, fpx)
    u0, v0, r = circle_from_points(uv)
    return features_from_circle(u0, v0, r), uv

def feature_error(s, s_star):
    return s - s_star

# --- Interaction matrix ---
def apply_cam_twist_to_pose(pose_xyth, d_cam):
    x, y, th = pose_xyth
    dx_c, dy_c, dth = d_cam
    c, s = np.cos(th), np.sin(th)
    dx_w = c*dx_c - s*dy_c
    dy_w = s*dx_c + c*dy_c
    return np.array([x + dx_w, y + dy_w, wrap_angle(th + dth)])

def numeric_interaction_matrix_3dof(pose_xyth, rim_w, fpx, eps_t=1e-4):
    s0, _ = get_features(pose_xyth, rim_w, fpx)
    L = np.zeros((s0.size, 2))  # yaw는 별도 제어하므로 2열만
    deltas = [np.array([eps_t,0,0]), np.array([0,eps_t,0])]
    for j, d in enumerate(deltas):
        pose_p = apply_cam_twist_to_pose(pose_xyth, d)
        s1, _ = get_features(pose_p, rim_w, fpx)
        ds = s1 - s0
        step = eps_t
        L[:, j] = ds / step
    return L

# --- Scenario (rim target) ---
rim_center_world = np.array([0.0, 0.0, 2.5])
rim_world = make_rim_world(rim_center_world, radius=0.5, tilt_rx_deg=20.0, tilt_ry_deg=5.0, n=200)
desired_pose = np.array([0.0, 0.0, 0.0])
desired_features, desired_uv = get_features(desired_pose, rim_world, FOCAL_LENGTH)

# 초기 pose
np.random.seed()
camera_pose = np.array([
    np.random.uniform(-3.0, 3.0),
    np.random.uniform(-1.0, 4.0),
    np.random.uniform(-np.pi, np.pi)
])

# --- 기록 ---
pose_history = [camera_pose.copy()]
error_history = []
feature_history = []
feat_err_components = []

# --- Main loop ---
for step in range(TOTAL_STEPS):
    s, uv = get_features(camera_pose, rim_world, FOCAL_LENGTH)
    feature_history.append(uv)

    e = feature_error(s, desired_features)
    yaw_err = wrap_angle(camera_pose[2] - desired_pose[2])
    error_norm = np.linalg.norm(e) + abs(yaw_err)

    error_history.append(error_norm)
    feat_err_components.append(np.hstack([np.abs(e), abs(yaw_err)]))

    current_lambda = max(LAMBDA_MIN, min(LAMBDA_MAX, LAMBDA_MAX*(error_norm/50.0)))

    if error_norm < 1.0:
        print(f"목표 도달! {step*DT:.2f} 초 소요.")
        break

    # translation IBVS
    L = numeric_interaction_matrix_3dof(camera_pose, rim_world, FOCAL_LENGTH)
    v_trans = -current_lambda * (np.linalg.pinv(L) @ e)

    # yaw control (separate P control)
    v_yaw = -K_YAW * yaw_err

    # update pose
    x, y, th = camera_pose
    c, s = np.cos(th), np.sin(th)
    vx_w = c*v_trans[0] - s*v_trans[1]
    vy_w = s*v_trans[0] + c*v_trans[1]
    camera_pose = np.array([x + vx_w*DT, y + vy_w*DT, wrap_angle(th + v_yaw*DT)])
    pose_history.append(camera_pose.copy())

pose_history = np.array(pose_history)
error_history = np.array(error_history)
feature_history = np.array(feature_history, dtype=object)
feat_err_components = np.array(feat_err_components)

# --- 시각화 ---
# fig = plt.figure(figsize=(16, 8))
fig = plt.figure(figsize=(20, 12), dpi=150)
ax1 = fig.add_subplot(2, 2, 1) # World view
ax2 = fig.add_subplot(2, 2, 2) # Camera view
ax3 = fig.add_subplot(2, 1, 2) # Error plot

# 1. World View (ax1) 설정
ax1.set_title("World View (Top-down)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")

ax1.grid(True); ax1.set_aspect('equal')

# World View axis를 trajectory에 맞춰 자동 설정
x_min, x_max = np.min(pose_history[:,0]), np.max(pose_history[:,0])
y_min, y_max = np.min(pose_history[:,1]), np.max(pose_history[:,1])

# 여유 margin 주기 (10%)
margin_x = 0.1 * (x_max - x_min if x_max > x_min else 1.0)
margin_y = 0.1 * (y_max - y_min if y_max > y_min else 1.0)

ax1.set_xlim(x_min - margin_x, x_max + margin_x)
ax1.set_ylim(y_min - margin_y, y_max + margin_y)

peg_radius = 0.5  # rim 반지름

# Rim (고정 target hole)
theta = np.linspace(0, 2*np.pi, 200)
rim_x = rim_center_world[0] + peg_radius*np.cos(theta)
rim_y = rim_center_world[1] + peg_radius*np.sin(theta)
ax1.plot(rim_x, rim_y, 'g-', lw=2, label="Rim (Target Hole)")

# Rim center marker
ax1.plot(rim_center_world[0], rim_center_world[1], 'g*', markersize=15, label="Rim Center")

# 전체 경로 (점선)
ax1.plot(pose_history[:,0], pose_history[:,1], 'b--', lw=1, alpha=0.5, label="Full Path")

# 현재 경로와 카메라 위치 핸들
current_path_plot, = ax1.plot([], [], 'b-', lw=2, label="Robot Path")
camera_arrow = ax1.arrow(0, 0, 0.1, 0.1, head_width=0.15, fc='r', ec='r', label="Camera Pose")

# Peg body (gripper가 들고 있는 원통) 핸들
peg_body_plot, = ax1.plot([], [], 'm-', lw=2, label="Gripper Peg")

ax1.legend()
ax1.axis([-3, 3, -1, 4])


# 2. Camera View (ax2) 설정
ax2.set_title("Camera View (Ellipse Projection)")
ax2.set_xlabel("u (pixels)")
ax2.set_ylabel("v (pixels)")
ax2.grid(True); ax2.set_aspect('equal')
# 목표 타원은 고정이므로 한 번만 그림
des_uv, _ = project_points_to_image(rim_world, desired_pose, FOCAL_LENGTH)
ax2.plot(des_uv[:,0], des_uv[:,1], 'r.', markersize=2, label="Desired Rim (s*)")
# 현재 타원을 업데이트할 artist 핸들을 미리 생성
current_rim_plot, = ax2.plot([], [], 'b.', markersize=2, label="Current Rim (s)")
ax2.legend()
ax2_window_size = 200
ax2.axis([-ax2_window_size, ax2_window_size, -ax2_window_size, ax2_window_size])


# 3. Error Plot (ax3) 설정
ax3.set_title("Feature Error Norm & Components")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Error")
ax3.grid(True)
# times = np.arange(len(pose_history)) * DT
times = np.arange(len(error_history)) * DT
# 에러 그래프들을 그리고 핸들을 변수에 저장
line_err_norm, = ax3.plot(times, error_history, label="||e||")
line_err_u, = ax3.plot(times, feat_err_components[:,0], label="|u0 - u0*| [px]")
line_err_v, = ax3.plot(times, feat_err_components[:,1], label="|v0 - v0*| [px]")
ax3.legend()
# Y축 범위는 전체 에러 기록을 바탕으로 설정
ax3.set_ylim(0, max(1.0, np.max(error_history)*1.1))


# 애니메이션 함수 정의
def animate(i):
    # World View 업데이트
    current_path_plot.set_data(pose_history[:i+1, 0], pose_history[:i+1, 1])
    x, y, th = pose_history[i]

    # 카메라 방향 화살표
    camera_arrow.set_data(x=x, y=y, dx=0.3*np.cos(th), dy=0.3*np.sin(th))

    # Peg body (gripper가 들고 있는 원통) 업데이트
    peg_x = x + peg_radius*np.cos(theta)
    peg_y = y + peg_radius*np.sin(theta)
    peg_body_plot.set_data(peg_x, peg_y)

    # Camera View 업데이트
    if i < len(feature_history):
        uv_i = feature_history[i]
        current_rim_plot.set_data(uv_i[:,0], uv_i[:,1])

    # Error Plot 업데이트
    t_i = times[:i+1]
    line_err_norm.set_data(t_i, error_history[:i+1])
    line_err_u.set_data(t_i, feat_err_components[:i+1, 0])
    line_err_v.set_data(t_i, feat_err_components[:i+1, 1])
    line_err_area.set_data(t_i, feat_err_components[:i+1, 2])

    return (current_path_plot, camera_arrow, peg_body_plot,
            current_rim_plot, line_err_norm, line_err_u,
            line_err_v, line_err_area)

# 애니메이션 실행
ani = FuncAnimation(fig, animate, frames=len(pose_history), interval=DT*1000, 
                    blit=True, repeat=False)

print("애니메이션을 'ibvs_ellipse_schedule.mp4' 로 저장을 시도합니다...")
try:
    sim_duration = len(pose_history) * DT     # 시뮬레이션 실제 시간 (초)
    fps = int(len(pose_history) / sim_duration)  # 초당 프레임 수 (≈ 1/DT)

    ani.save('ibvs_ellipse_schedule.mp4',
            writer='ffmpeg',
            fps=fps, dpi=100)
    print("저장이 완료되었습니다.")
except Exception as ex:
    print(f"[Info] mp4 저장 실패 (ffmpeg 미설치 등): {ex}")

plt.tight_layout()
plt.show()
