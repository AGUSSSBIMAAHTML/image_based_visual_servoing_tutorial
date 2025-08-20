import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Params ---
LAMBDA = 2
DT = 0.1
SIM_TIME = 20
TOTAL_STEPS = int(SIM_TIME / DT)
FOCAL_LENGTH = 500.0

# --- Helpers (rotations) ---
def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def Rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

# --- Geometry: circle (rim) in world, projected to image as ellipse ---
def make_rim_world(center_w, radius=0.5, tilt_rx_deg=20.0, tilt_ry_deg=5.0, n=200):
    rx = np.deg2rad(tilt_rx_deg)
    ry = np.deg2rad(tilt_ry_deg)
    # tilt plane: Ry * Rx
    c,s = np.cos, np.sin
    Rx = np.array([[1,0,0],[0,c(rx),-s(rx)],[0,s(rx),c(rx)]])
    Ry = np.array([[c(ry),0,s(ry)],[0,1,0],[-s(ry),0,c(ry)]])
    R_plane = Ry @ Rx
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring_local = np.stack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)], axis=1)
    pts_w = (R_plane @ ring_local.T).T + center_w
    return pts_w  # (N,3)

def project_points_to_image(pts_w, pose_xyth, fpx):
    """
    pose_xyth = [x, y, theta], 카메라는 z축을 보는 pinhole, 피처 깊이 Z는 pts_w[:,2] 그대로 사용(상수 가정)
    월드→카메라: yaw만; 카메라 z축이 월드 z축과 일치한다고 가정
    """
    x, y, th = pose_xyth
    Rcw = Rz(th).T
    tcw = -Rcw @ np.array([x, y, 0.0])  # z 이동은 안 씀(깊이 고정)
    pts_c = (Rcw @ pts_w.T).T + tcw
    X, Y, Z = pts_c[:,0], pts_c[:,1], pts_c[:,2]  # 여기서 Z는 world Z가 yaw 회전만 거친 값 → 상수 수준
    u = fpx * (X / Z)
    v = fpx * (Y / Z)
    return np.stack([u, v], axis=1), Z

def ellipse_from_points(uv):
    """타원 파라미터 근사: 중심, 주/부 반지름(a,b ~ 표준편차 스케일), 기울기 alpha"""
    u0, v0 = uv.mean(axis=0)
    duv = uv - np.array([u0, v0])
    C = (duv.T @ duv) / max(uv.shape[0], 1)
    evals, evecs = np.linalg.eigh(C)  # 작은->큰
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]; evecs = evecs[:, idx]
    a = np.sqrt(max(evals[0], 1e-12))
    b = np.sqrt(max(evals[1], 1e-12))
    alpha = wrap_angle(np.arctan2(evecs[1,0], evecs[0,0]))
    return float(u0), float(v0), float(a), float(b), float(alpha)

def features_from_ellipse(u0, v0, a, b, alpha):
    # s = [u0, v0, log(a*b), alpha]
    return np.array([u0, v0, np.log(max(a*b, 1e-12)), alpha])

def get_features(pose_xyth, rim_w, fpx):
    uv, _ = project_points_to_image(rim_w, pose_xyth, fpx)
    u0, v0, a, b, alpha = ellipse_from_points(uv)
    return features_from_ellipse(u0, v0, a, b, alpha), uv

def feature_error(s, s_star):
    e = s - s_star
    e[-1] = wrap_angle(e[-1])  # alpha 차이는 각도 wrap
    return e

# --- Numeric L for 3-DoF (vx, vy, wz) ---
def apply_cam_twist_to_pose(pose_xyth, d_cam):
    """
    카메라 프레임의 작은 twist [dx_cam, dy_cam, dtheta]를
    월드 포즈 [x,y,theta]에 반영 (dt=1 근사)
    """
    x, y, th = pose_xyth
    dx_c, dy_c, dth = d_cam
    c, s = np.cos(th), np.sin(th)
    # cam->world 변환
    dx_w = c*dx_c - s*dy_c
    dy_w = s*dx_c + c*dy_c
    return np.array([x + dx_w, y + dy_w, wrap_angle(th + dth)])

def numeric_interaction_matrix_3dof(pose_xyth, rim_w, fpx, eps_t=1e-4, eps_r=1e-4):
    """
    L ≈ ds/d[vx, vy, wz]  (z 고정)
    작은 카메라 프레임 변위를 포즈에 적용하고 특성 변화율로 L의 각 열 근사
    """
    s0, _ = get_features(pose_xyth, rim_w, fpx)
    L = np.zeros((s0.size, 3))
    deltas = [
        np.array([ eps_t, 0.0,   0.0]),  # +vx
        np.array([ 0.0,   eps_t, 0.0]),  # +vy
        np.array([ 0.0,   0.0,   eps_r]) # +wz
    ]
    for j, d in enumerate(deltas):
        pose_p = apply_cam_twist_to_pose(pose_xyth, d)
        s1, _ = get_features(pose_p, rim_w, fpx)
        ds = s1 - s0
        ds[-1] = wrap_angle(ds[-1])
        step = d[j] if j < 2 else eps_r
        L[:, j] = ds / step
    return L

# --- Scenario (rim target) ---
# 원형 림 중심과 깊이는 기존 코드의 Z=2.5를 그대로 사용
rim_center_world = np.array([0.0, 0.0, 2.5])
rim_world = make_rim_world(rim_center_world, radius=0.5, tilt_rx_deg=20.0, tilt_ry_deg=5.0, n=200)

# 원하는(타겟) 피처: 카메라가 정면(yaw=0), x=y=0일 때의 타원
desired_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta], z는 고정이므로 없음
desired_features, desired_uv = get_features(desired_pose, rim_world, FOCAL_LENGTH)

# 초기 포즈는 기존과 유사
camera_pose = np.array([-2.0, 2.0, np.deg2rad(45.0)])

# --- 기록 ---
pose_history = [camera_pose.copy()]
error_history = []      # ||e||2
feature_history = []    # uv rim 샘플 찍기용
feat_err_components = []  # |e_u0|, |e_v0|, |e_logab|, |e_alpha|

# --- 메인 루프 ---
for step in range(TOTAL_STEPS):
    # 센싱: ellipse 피처
    s, uv = get_features(camera_pose, rim_world, FOCAL_LENGTH)
    feature_history.append(uv)

    # 에러
    e = feature_error(s, desired_features)
    error_history.append(np.linalg.norm(e))
    feat_err_components.append(np.array([abs(e[0]), abs(e[1]), abs(e[2]), abs(e[3])]))

    if np.linalg.norm(e) < 1.0:
        print(f"목표 도달! {step*DT:.2f} 초 소요.")
        break

    # 자코비안 (수치미분)
    L = numeric_interaction_matrix_3dof(camera_pose, rim_world, FOCAL_LENGTH)

    # 제어법칙: v_cam = -λ * L^+ * e
    L_pinv = np.linalg.pinv(L)
    v_cam = -LAMBDA * (L_pinv @ e)  # [vx, vy, wz] in camera frame

    # 포즈 적분: 카메라→월드 변환 후 x,y만, yaw는 그대로
    x, y, th = camera_pose
    c, s = np.cos(th), np.sin(th)
    vx_w = c*v_cam[0] - s*v_cam[1]
    vy_w = s*v_cam[0] + c*v_cam[1]
    dth  = v_cam[2]

    camera_pose = np.array([x + vx_w*DT, y + vy_w*DT, wrap_angle(th + dth*DT)])
    pose_history.append(camera_pose.copy())

pose_history = np.array(pose_history)
error_history = np.array(error_history)
feature_history = np.array(feature_history, dtype=object)  # list of arrays
feat_err_components = np.array(feat_err_components)

# --- 시각화 ---
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 1, 2)

# World view
ax1.set_title("World View (Top-down)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.grid(True); ax1.set_aspect('equal')
ax1.plot(rim_center_world[0], rim_center_world[1], 'g*', markersize=15, label="Rim Center (World)")
ax1.plot(pose_history[:,0], pose_history[:,1], 'b-', lw=2, label="Robot Path")
ax1.legend()
ax1.axis([-3, 3, -1, 4])

# Camera view: current rim & desired rim
ax2.set_title("Camera View (Ellipse Projection)")
ax2.set_xlabel("u (pixels)")
ax2.set_ylabel("v (pixels)")
ax2.grid(True); ax2.set_aspect('equal')
# desired rim under desired pose (정면)
des_uv, _ = project_points_to_image(rim_world, desired_pose, FOCAL_LENGTH)
ax2.plot(des_uv[:,0], des_uv[:,1], 'r.', markersize=2, label="Desired Rim (s*)")

# --- 기존: 마지막 스텝 파란 원을 미리 그림 (누적 원인)
# cur_uv, _ = project_points_to_image(rim_world, pose_history[-1], FOCAL_LENGTH)
# ax2.plot(cur_uv[:,0], cur_uv[:,1], 'b.', markersize=2, label="Current Rim (s)")

# --- 변경: 빈 핸들 하나 만들어 두고 매 프레임 set_data 로 업데이트
current_rim_plot, = ax2.plot([], [], 'b.', markersize=2, label="Current Rim (s)")
ax2.legend()
ax2_window_size = 800   # 픽셀 단위, 카메라 뷰의 가로/세로 크기 
ax2.axis([-ax2_window_size, ax2_window_size, -ax2_window_size, ax2_window_size])

# Error plot
ax3.set_title("Feature Error Norm & Components")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Error")
ax3.grid(True)
times = np.arange(error_history.shape[0]) * DT
ax3.plot(times, error_history, label="||e||")
if feat_err_components.size > 0:
    ax3.plot(times, feat_err_components[:len(times),0], label="|u0 - u0*| [px]")
    ax3.plot(times, feat_err_components[:len(times),1], label="|v0 - v0*| [px]")
    ax3.plot(times, feat_err_components[:len(times),2], label="|log(ab)-log(ab)*|")
    ax3.plot(times, np.rad2deg(feat_err_components[:len(times),3]), label="|alpha-alpha*| [deg]")
ax3.legend()

def animate(i):
    # World path + heading
    ax1.plot(pose_history[:i+1,0], pose_history[:i+1,1], 'b-', lw=2)
    x, y, th = pose_history[i]
    dx, dy = 0.2*np.cos(th), 0.2*np.sin(th)
    ax1.plot([x, x+dx], [y, y+dy], 'r-', lw=2)

    # Camera view animation
    uv_i, _ = project_points_to_image(rim_world, pose_history[i], FOCAL_LENGTH)
    ax2.plot(uv_i[:,0], uv_i[:,1], 'b.', markersize=2)

    # Error update
    t = np.arange(i+1)*DT
    ax3.lines[0].set_data(t, error_history[:i+1])
    ax3.set_xlim(0, SIM_TIME)
    ax3.set_ylim(0, max(1.0, np.max(error_history)*1.1))
    return ax1, ax2, ax3

ani = FuncAnimation(fig, animate, frames=len(pose_history), interval=DT*1000, blit=False, repeat=False)

print("애니메이션을 'ibvs_ellipse_no_schedule.mp4' 로 저장을 시도합니다...")
try:
    ani.save('ibvs_ellipse_no_schedule.mp4', writer='ffmpeg', fps=30, dpi=100)
    print("저장이 완료되었습니다.")
except Exception as ex:
    print(f"[Info] mp4 저장 실패 (ffmpeg 미설치 등): {ex}")

plt.tight_layout()
plt.show()
