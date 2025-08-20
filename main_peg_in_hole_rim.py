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
# [Ref] Gain λ는 ė = -λ e 의 지수수렴 설계 아이디어에서 옴 (Sec. 34.1, Eq. (34.3)→(34.4)).
#      Adaptive gain은 본문에 정식 공식은 없지만, 실무적으로 λ 튜닝/변조는 흔함 (Sec. 34.1 논의).

# --- Helpers (rotations) ---
def wrap_angle(a: float) -> float:
    # [Ref] 각도 wrap은 구현 디테일. 제어 이론상 e를 안정적으로 줄이기 위한 수단 (Sec. 34.1 개요).
    return (a + np.pi) % (2*np.pi) - np.pi

def Rz(yaw: float) -> np.ndarray:
    # [Ref] 카메라/월드 좌표 변환의 일부. VC = (v_c, ω_c)와 ṡ = L_s v_c 관계를 구현할 때 필수 (Eq. (34.2)).
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

# --- Geometry ---
def make_rim_world(center_w, radius=0.5, tilt_rx_deg=20.0, tilt_ry_deg=5.0, n=200):
    # [Ref] 타원/원 등 기하 프리미티브의 사용은 IBVS의 일반화된 피처 설계와 맞닿음 (Sec. 34.2.7).
    rx = np.deg2rad(tilt_rx_deg); ry = np.deg2rad(tilt_ry_deg)
    c,s = np.cos, np.sin
    Rx = np.array([[1,0,0],[0,c(rx),-s(rx)],[0,s(rx),c(rx)]])
    Ry = np.array([[c(ry),0,s(ry)],[0,1,0],[-s(ry),0,c(ry)]])
    R_plane = Ry @ Rx
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring_local = np.stack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)], axis=1)
    return (R_plane @ ring_local.T).T + center_w

def project_points_to_image(pts_w, pose_xyth, fpx):
    # [Ref] 투영 모델: x = X/Z, y = Y/Z (정규화 좌표) → u = f x, v = f y.
    #       이로부터 ẋ, ẏ 와 v_c 의 선형관계 ṡ = L_s v_c 유도 (Eq. (34.2), (34.11)~(34.12)).
    x, y, th = pose_xyth
    Rcw = Rz(th).T
    tcw = -Rcw @ np.array([x, y, 0.0])
    pts_c = (Rcw @ pts_w.T).T + tcw
    X, Y, Z = pts_c[:,0], pts_c[:,1], pts_c[:,2]
    return np.stack([fpx*(X/Z), fpx*(Y/Z)], axis=1), Z

# --- Features (circle only) ---
def circle_from_points(uv):
    # [Ref] 원/타원 컨투어로부터 (중심, 반경) 같은 모멘트/기하 피처 사용은 Sec. 34.2.7에 해당.
    u0, v0 = uv.mean(axis=0)
    duv = uv - np.array([u0, v0])
    r = np.sqrt(np.mean(np.sum(duv**2, axis=1)))
    return float(u0), float(v0), float(r)

def features_from_circle(u0, v0, r):
    # [Ref] s = (u0, v0, log r^2) 형태의 피처 벡터. e = s - s* 가 기본 (Eq. (34.1)).
    return np.array([u0, v0, np.log(max(r*r, 1e-12))])

def get_features(pose_xyth, rim_w, fpx):
    # [Ref] m(t) (원 픽셀 샘플) → s(m(t), a) (피처 벡터) 매핑 (Sec. 34.1, Eq. (34.1) 정의).
    uv, _ = project_points_to_image(rim_w, pose_xyth, fpx)
    u0, v0, r = circle_from_points(uv)
    return features_from_circle(u0, v0, r), uv

def feature_error(s, s_star):
    # [Ref] e(t) = s(m(t), a) - s* (Eq. (34.1)).
    return s - s_star

# --- Interaction matrix ---
def apply_cam_twist_to_pose(pose_xyth, d_cam):
    # [Ref] 카메라 프레임의 미소 변위 d_cam = (dx_c, dy_c, dθ) 를 월드로 적분.
    #       ṡ = L_s v_c 의 v_c 적용을 수치적으로 흉내내는 구성 (Eq. (34.2)).
    x, y, th = pose_xyth
    dx_c, dy_c, dth = d_cam
    c, s = np.cos(th), np.sin(th)
    dx_w = c*dx_c - s*dy_c
    dy_w = s*dx_c + c*dy_c
    return np.array([x + dx_w, y + dy_w, wrap_angle(th + dth)])

def numeric_interaction_matrix_3dof(pose_xyth, rim_w, fpx, eps_t=1e-4):
    # [Ref] L_s 를 해석식 대신 유한차분으로 근사 (수치적 L̂_e). 실제 시스템에서 L 또는 L^+의 추정/근사 필요 (Sec. 34.1,
    #       Eq. (34.4)~(34.6) 주변 논의; Sec. 34.2.2 Approximating the Interaction Matrix).
    s0, _ = get_features(pose_xyth, rim_w, fpx)
    L = np.zeros((s0.size, 2))  # yaw는 별도 제어하므로 2열만 (v_x, v_y 대응)
    deltas = [np.array([eps_t,0,0]), np.array([0,eps_t,0])]
    for j, d in enumerate(deltas):
        pose_p = apply_cam_twist_to_pose(pose_xyth, d)
        s1, _ = get_features(pose_p, rim_w, fpx)
        ds = s1 - s0
        step = eps_t
        L[:, j] = ds / step
    return L
    # [Ref] 이상적인 해석적 L_x 는 Eq. (34.12) (점 피처 기준)처럼 주어지며, 깊이 Z 추정이 필요.
    #       여기서는 원-모멘트 피처라 정확한 폐식은 생략하고 수치 근사로 대체 (Sec. 34.2.7 + 34.2.2).

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
    # [Ref] 루프에서 m(t)→s(t) 계산, e = s - s* (Eq. (34.1)).
    s, uv = get_features(camera_pose, rim_world, FOCAL_LENGTH)
    feature_history.append(uv)

    e = feature_error(s, desired_features)
    yaw_err = wrap_angle(camera_pose[2] - desired_pose[2])

    # [Ref] ė = L_e v_c, ė = -λ e를 원하면 v_c = -λ L_e^+ e (Eq. (34.2)~(34.4)).
    error_norm = np.linalg.norm(e) + abs(yaw_err)
    error_history.append(error_norm)
    feat_err_components.append(np.hstack([np.abs(e), abs(yaw_err)]))

    # [Ref] λ 스케줄링(휴리스틱): 안정성/응답 타협 (Lyapunov 기반 안정성 논의는 Eq. (34.6) 참조).
    current_lambda = max(LAMBDA_MIN, min(LAMBDA_MAX, LAMBDA_MAX*(error_norm/50.0)))

    if error_norm < 1.0:
        print(f"목표 도달! {step*DT:.2f} 초 소요.")
        break

    # translation IBVS
    # [Ref] 수치 근사 L̂_e 로 v_trans = -λ L̂_e^+ e 계산 (Eq. (34.4), (34.5)).
    L = numeric_interaction_matrix_3dof(camera_pose, rim_world, FOCAL_LENGTH)
    v_trans = -current_lambda * (np.linalg.pinv(L) @ e)

    # yaw control (separate P control)
    # [Ref] 회전은 PBVS식 제어(ω_c = -λ θ u; Eq. (34.27))를 IBVS 번역축과 
    # 하이브리드로 분리하는 2.5D/Hybrid 아이디어 (Sec. 34.4.1, Eq. (34.27)~(34.28)).
    v_yaw = -K_YAW * yaw_err

    # update pose
    # [Ref] v_trans 는 카메라 좌표계 속도이므로 월드로 변환해 적분 (ṡ = L_s v_c의 v_c 적용).
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
# [Ref] Fig. 34.1~34.5 등은 이미지/카메라/경로 동작 예시를 보여줌. 아래 플롯은 유사한 해석을 돕기 위한 것.
fig = plt.figure(figsize=(20, 12), dpi=150)
ax1 = fig.add_subplot(2, 2, 1) # World view
ax2 = fig.add_subplot(2, 2, 2) # Camera view
ax3 = fig.add_subplot(2, 1, 2) # Error plot

# 1. World View (ax1) 설정
ax1.set_title("World View (Top-down)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.grid(True); ax1.set_aspect('equal')

# 자동 축 설정
x_min, x_max = np.min(pose_history[:,0]), np.max(pose_history[:,0])
y_min, y_max = np.min(pose_history[:,1]), np.max(pose_history[:,1])
margin_x = 0.1 * (x_max - x_min if x_max > x_min else 1.0)
margin_y = 0.1 * (y_max - y_min if y_max > y_min else 1.0)
ax1.set_xlim(x_min - margin_x, x_max + margin_x)
ax1.set_ylim(y_min - margin_y, y_max + margin_y)

peg_radius = 0.5  # rim 반지름
theta = np.linspace(0, 2*np.pi, 200)
rim_x = rim_center_world[0] + peg_radius*np.cos(theta)
rim_y = rim_center_world[1] + peg_radius*np.sin(theta)
ax1.plot(rim_x, rim_y, 'g-', lw=2, label="Rim (Target Hole)")
ax1.plot(rim_center_world[0], rim_center_world[1], 'g*', markersize=15, label="Rim Center")
ax1.plot(pose_history[:,0], pose_history[:,1], 'b--', lw=1, alpha=0.5, label="Full Path")

# 현재 경로와 카메라 위치 핸들
current_path_plot, = ax1.plot([], [], 'b-', lw=2, label="Robot Path")
camera_arrow = ax1.arrow(0, 0, 0.1, 0.1, head_width=0.15, fc='r', ec='r', label="Camera Pose")  # 주: Arrow는 set_data 미지원

# Peg body
peg_body_plot, = ax1.plot([], [], 'm-', lw=2, label="Gripper Peg")
ax1.legend()
ax1.axis([-3, 3, -1, 4])

# 2. Camera View (ax2) 설정
ax2.set_title("Camera View (Ellipse Projection)")
ax2.set_xlabel("u (pixels)")
ax2.set_ylabel("v (pixels)")
ax2.grid(True); ax2.set_aspect('equal')
# [Ref] IBVS에서 이미지 상 목표 궤적을 보는 것은 Fig. 34.2~34.5의 관점과 동일.
des_uv, _ = project_points_to_image(rim_world, desired_pose, FOCAL_LENGTH)
ax2.plot(des_uv[:,0], des_uv[:,1], 'r.', markersize=2, label="Desired Rim (s*)")
current_rim_plot, = ax2.plot([], [], 'b.', markersize=2, label="Current Rim (s)")
ax2.legend()
ax2_window_size = 200
ax2.axis([-ax2_window_size, ax2_window_size, -ax2_window_size, ax2_window_size])

# 3. Error Plot (ax3) 설정
ax3.set_title("Feature Error Norm & Components")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Error")
ax3.grid(True)
times = np.arange(len(error_history)) * DT
line_err_norm, = ax3.plot(times, error_history, label="||e||")
line_err_u, = ax3.plot(times, feat_err_components[:,0], label="|u0 - u0*| [px]")
line_err_v, = ax3.plot(times, feat_err_components[:,1], label="|v0 - v0*| [px]")
ax3.legend()
ax3.set_ylim(0, max(1.0, np.max(error_history)*1.1))

# 애니메이션 함수
def animate(i):
    # [Ref] 이미지 궤적/카메라 경로의 시각적 해석은 Fig. 34.2~34.5와 동일한 교육적 목적.
    current_path_plot.set_data(pose_history[:i+1, 0], pose_history[:i+1, 1])
    x, y, th = pose_history[i]

    # NOTE: Arrow 객체는 set_data가 없어 재생성 또는 FancyArrowPatch 사용이 보통임.
    # 여기서는 단순 재그리기 대신 참고용 주석만 남김.
    # (실사용시: camera_arrow.remove(); camera_arrow = ax1.arrow(...))

    peg_x = x + peg_radius*np.cos(theta)
    peg_y = y + peg_radius*np.sin(theta)
    peg_body_plot.set_data(peg_x, peg_y)

    if i < len(feature_history):
        uv_i = feature_history[i]
        current_rim_plot.set_data(uv_i[:,0], uv_i[:,1])

    t_i = times[:i+1]
    line_err_norm.set_data(t_i, error_history[:i+1])
    line_err_u.set_data(t_i, feat_err_components[:i+1, 0])
    line_err_v.set_data(t_i, feat_err_components[:i+1, 1])

    return (current_path_plot, peg_body_plot,
            current_rim_plot, line_err_norm, line_err_u, line_err_v)

ani = FuncAnimation(fig, animate, frames=len(pose_history), interval=DT*1000, 
                    blit=True, repeat=False)

video_filename = 'ibvs_rim_schedule.mp4'
print(f"Trying to save '{video_filename}' ...")
try:
    sim_duration = len(pose_history) * DT
    fps = int(len(pose_history) / sim_duration)  # ≈ 1/DT

    ani.save(video_filename, writer='ffmpeg', fps=fps, dpi=100)
    print("저장이 완료되었습니다.")
except Exception as ex:
    print(f"[Info] mp4 저장 실패 (ffmpeg 미설치 등): {ex}")

plt.tight_layout()
plt.show()
