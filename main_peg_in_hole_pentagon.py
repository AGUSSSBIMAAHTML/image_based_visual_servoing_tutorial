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
# [Ref] λ 게인: ė = -λ e 형태의 지수수렴 설계 (Sec. 34.1, Eq. (34.3)~(34.4)).
#      적응/스케줄형 λ는 실무적 튜닝에 해당 (Sec. 34.1 논의).

# --- Helpers (rotations) ---
def wrap_angle(a: float) -> float:
    # [Ref] 각도 wrap은 구현 디테일. 오차 정의 e = s - s* 의 안정적 감소에 도움 (Sec. 34.1).
    return (a + np.pi) % (2*np.pi) - np.pi

def Rz(yaw: float) -> np.ndarray:
    # [Ref] 좌표계 회전. 카메라 twist v_c와 ṡ = L_s v_c 연결을 구현할 때 필수 (Eq. (34.2)).
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def Rz2(yaw: float) -> np.ndarray:
    # [Ref] 2D 회전 행렬 (world top-down 표시용). 시각화 유틸.
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s],
                     [s,  c]])

# --- Geometry: tilted plane N-gon in world ---
def make_ngon_world(center_w, radius=0.5, tilt_rx_deg=20.0, tilt_ry_deg=5.0, n_sides=5):
    """Return N polygon vertices on a tilted plane around center_w."""
    # [Ref] 점/선/모멘트 등 다양한 피처 사용 가능 (Sec. 34.2.7).
    rx = np.deg2rad(tilt_rx_deg); ry = np.deg2rad(tilt_ry_deg)
    c,s = np.cos, np.sin
    Rx = np.array([[1,0,0],[0,c(rx),-s(rx)],[0,s(rx),c(rx)]])
    Ry = np.array([[c(ry),0,s(ry)],[0,1,0],[-s(ry),0,c(ry)]])
    R_plane = Ry @ Rx
    t = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    poly_local = np.stack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)], axis=1)
    return (R_plane @ poly_local.T).T + center_w  # (N,3)

def project_points_to_image(pts_w, pose_xyth, fpx):
    # [Ref] 핀홀 투영: (x = X/Z, y = Y/Z) → (u = f x, v = f y).
    #      점 피처 s = [u,v,…]의 시공간 변화는 ṡ = L_s v_c 로 표현 (Eq. (34.2), (34.11)~(34.12)).
    x, y, th = pose_xyth
    Rcw = Rz(th).T
    tcw = -Rcw @ np.array([x, y, 0.0])
    pts_c = (Rcw @ pts_w.T).T + tcw
    X, Y, Z = pts_c[:,0], pts_c[:,1], pts_c[:,2]
    uv = np.stack([fpx*(X/Z), fpx*(Y/Z)], axis=1)
    return uv, Z

# --- Features (pentagon: 5 vertices -> 10D feature) ---
def features_from_polygon(uv):
    # [Ref] 점 피처를 직접 스택한 s ∈ R^{2N}. 고전적 IBVS 점 피처와 
    # 동일 철학 (Sec. 34.2.1; Eq. (34.11)~(34.12) 참조).
    return uv.reshape(-1)

def get_features(pose_xyth, poly_w, fpx):
    # [Ref] m(t)→s(t) 매핑: s = s(m(t), a) (Sec. 34.1, Eq. (34.1)).
    uv, _ = project_points_to_image(poly_w, pose_xyth, fpx)
    s = features_from_polygon(uv)
    return s, uv

def feature_error(s, s_star):
    # [Ref] e = s - s* (Sec. 34.1, Eq. (34.1)).
    return s - s_star

# --- Pose update helper (camera twist in camera frame) ---
def apply_cam_twist_to_pose(pose_xyth, d_cam):
    # [Ref] 카메라 프레임 미소변위 d_cam=(dx_c,dy_c,dθ) 적용.
    #      ṡ = L_s v_c 에서 v_c를 수치적으로 적분하는 역할 (Eq. (34.2)).
    x, y, th = pose_xyth
    dx_c, dy_c, dth = d_cam
    c, s = np.cos(th), np.sin(th)
    dx_w = c*dx_c - s*dy_c
    dy_w = s*dx_c + c*dy_c
    return np.array([x + dx_w, y + dy_w, wrap_angle(th + dth)])

# --- Numeric interaction matrix for translation (2 DOF: x/y in cam frame) ---
def numeric_interaction_matrix_3dof(pose_xyth, poly_w, fpx, eps_t=1e-4):
    # [Ref] 상호작용행렬 L_s를 유한차분으로 근사한 L̂ (Sec. 34.2.2; also 주변 논의).
    #      점 피처에 대한 해석 L은 Eq. (34.12)에 예시. 여기선 다점 스택이라 수치 근사 사용.
    s0, _ = get_features(pose_xyth, poly_w, fpx)     # dim = 2*N
    L = np.zeros((s0.size, 2))                       # yaw 별도 제어 → 번역 2자유도만
    deltas = [np.array([eps_t,0,0]), np.array([0,eps_t,0])]
    for j, d in enumerate(deltas):
        pose_p = apply_cam_twist_to_pose(pose_xyth, d)
        s1, _ = get_features(pose_p, poly_w, fpx)
        ds = s1 - s0
        L[:, j] = ds / eps_t
    return L

# --- Scenario (pentagon target) ---
rim_center_world = np.array([0.0, 0.0, 2.5])
n_sides = 5
poly_world = make_ngon_world(rim_center_world, radius=0.5, tilt_rx_deg=20.0, tilt_ry_deg=5.0, n_sides=n_sides)
desired_pose = np.array([0.0, 0.0, 0.0])
desired_features, desired_uv = get_features(desired_pose, poly_world, FOCAL_LENGTH)

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
feat_err_components = []   # [mean|duv|, max|duv|, |yaw_err|]

# --- Main loop ---
for step in range(TOTAL_STEPS):
    # [Ref] 루프에서 s(t), e(t) 업데이트 (Eq. (34.1)).
    s, uv = get_features(camera_pose, poly_world, FOCAL_LENGTH)

    e = feature_error(s, desired_features)  # (10,)
    yaw_err = wrap_angle(camera_pose[2] - desired_pose[2])

    # [Ref] duv 기반 개별 점 오차 통계(시각화용). 제어법칙에는 e만 사용.
    duv = (uv - desired_uv)                  # (5,2)
    per_pt = np.linalg.norm(duv, axis=1)     # (5,)
    mean_abs = float(np.mean(per_pt))
    max_abs  = float(np.max(per_pt))

    # [Ref] 목표: ė = -λ e 가 되도록 v_c 선택 (Eq. (34.3)~(34.5)).
    error_norm = np.linalg.norm(e) + abs(yaw_err)
    error_history.append(error_norm)
    feat_err_components.append([mean_abs, max_abs, abs(yaw_err)])

    # [Ref] λ 스케줄링 (휴리스틱): 큰 오차일 때 빠르고, 작아지면 안정적 (Eq. (34.6) 안정성 논의 주변).
    current_lambda = max(LAMBDA_MIN, min(LAMBDA_MAX, LAMBDA_MAX*(error_norm/50.0)))

    if error_norm < 1.0:
        print(f"Goal reached! t = {step*DT:.2f} s")
        break

    # translation IBVS (x,y in cam frame)
    # [Ref] v_trans = -λ L̂^+ e (Eq. (34.4)-(34.5)). 여기서 L̂은 수치 근사.
    L = numeric_interaction_matrix_3dof(camera_pose, poly_world, FOCAL_LENGTH)
    v_trans = -current_lambda * (np.linalg.pinv(L) @ e)   # (2,)

    # yaw control (separate P control)
    # [Ref] 회전은 PBVS/2.5D 하이브리드 아이디어로 분리 제어 (Sec. 34.4.1, Eq. (34.27)~(34.28) 철학).
    v_yaw = -K_YAW * yaw_err

    # update pose
    # [Ref] v_trans 는 카메라 프레임 속도 → 월드 합성/적분 (Eq. (34.2) 적용 관점).
    x, y, th = camera_pose
    c, s_ = np.cos(th), np.sin(th)
    vx_w = c*v_trans[0] - s_*v_trans[1]
    vy_w = s_*v_trans[0] + c*v_trans[1]
    camera_pose = np.array([x + vx_w*DT, y + vy_w*DT, wrap_angle(th + v_yaw*DT)])
    pose_history.append(camera_pose.copy())

pose_history = np.array(pose_history)
error_history = np.array(error_history)
feat_err_components = np.array(feat_err_components)  # (T,3)

# --- Visualization: Left(World) | Right(Loss) ---
fig, (ax1, ax3) = plt.subplots(
    1, 2, figsize=(20, 10), dpi=150,
    gridspec_kw={'width_ratios':[1,1]}, constrained_layout=True
)

# 1) World View
ax1.set_title("World View (Top-down)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.grid(True); ax1.set_aspect('equal')

# [Ref] 세계 경로/자세 시각화는 교재 Fig. 34.2~34.5류의 해석과 유사한 교육 목적.
x_min, x_max = np.min(pose_history[:,0]), np.max(pose_history[:,0])
y_min, y_max = np.min(pose_history[:,1]), np.max(pose_history[:,1])

peg_radius = 0.5
theta_draw = np.linspace(0, 2*np.pi, n_sides+1)  # closed for drawing
target_x = rim_center_world[0] + peg_radius*np.cos(theta_draw)
target_y = rim_center_world[1] + peg_radius*np.sin(theta_draw)

x_min = min(x_min, np.min(target_x))
x_max = max(x_max, np.max(target_x))
y_min = min(y_min, np.min(target_y))
y_max = max(y_max, np.max(target_y))

margin_x = 0.1 * (x_max - x_min if x_max > x_min else 1.0)
margin_y = 0.1 * (y_max - y_min if y_max > y_min else 1.0)
margin = max(margin_x, margin_y)

ax1.set_xlim(x_min - margin, x_max + margin)
ax1.set_ylim(y_min - margin, y_max + margin)

# Target polygon (schematic)
rim_x = rim_center_world[0] + peg_radius*np.cos(theta_draw)
rim_y = rim_center_world[1] + peg_radius*np.sin(theta_draw)
ax1.plot(rim_x, rim_y, 'g-', lw=2, label="Target Polygon (schematic)")
ax1.plot(rim_center_world[0], rim_center_world[1], 'g*', markersize=15, label="Target Center")

# Full path (dashed), current path handle
ax1.plot(pose_history[:,0], pose_history[:,1], 'b--', lw=1, alpha=0.5, label="Full Path")
current_path_plot, = ax1.plot([], [], 'b-', lw=2, label="Robot Path", animated=True)

# Camera heading (quiver)
# [Ref] 카메라 yaw(heading) 시각화.
camera_quiver = ax1.quiver([0],[0],[0],[0],
                           angles='xy', scale_units='xy', scale=1.0,
                           color='r', label="Camera Pose", animated=True)

# Robot-held polygon (magenta) rotated by camera yaw
peg_local = np.stack([peg_radius*np.cos(theta_draw),
                      peg_radius*np.sin(theta_draw)], axis=1)  # (N+1, 2)
peg_poly_plot, = ax1.plot([], [], 'm-', lw=2, label="Gripper Polygon", animated=True)

ax1.legend()

# 2) Loss Plot (Right)
times = np.arange(len(error_history)) * DT
ax3.set_title("Feature Error & Yaw")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Error")
ax3.grid(True)
# [Ref] yaw 오차는 deg로 가독성 향상 (시각화 선택 사항).
feat_err_components[:,2]= np.rad2deg(feat_err_components[:,2])
line_err_norm, = ax3.plot(times, error_history, label="||e||")
line_err_mean, = ax3.plot(times, feat_err_components[:,0], label="Mean |Δuv| [px]")
line_err_max,  = ax3.plot(times, feat_err_components[:,1], label="Max |Δuv| [px]")
line_yaw_abs,  = ax3.plot(times, feat_err_components[:,2], label="|yaw_err| [deg]")
ax3.legend()
ax3.set_ylim(0, max(
    1.0,
    np.max([
        np.max(error_history),
        np.max(feat_err_components[:,0]),
        np.max(feat_err_components[:,1]),
        np.max(feat_err_components[:,2])  # deg
    ]) * 1.1
))

def animate(i):
    # [Ref] 이미지/경로 시간 변화 해석은 Fig. 34.2~34.5류와 같은 교육 목적.
    current_path_plot.set_data(pose_history[:i+1, 0], pose_history[:i+1, 1])

    # Camera pose (quiver)
    x, y, th = pose_history[i]
    camera_quiver.set_offsets([x, y])
    camera_quiver.set_UVC(0.4*np.cos(th), 0.4*np.sin(th))

    # Robot-held polygon rotated by yaw, translated to (x,y)
    R2 = Rz2(th)
    peg_world = peg_local @ R2.T + np.array([x, y])
    peg_poly_plot.set_data(peg_world[:,0], peg_world[:,1])

    # Loss plot update
    t_i = times[:i+1]
    line_err_norm.set_data(t_i, error_history[:i+1])
    line_err_mean.set_data(t_i, feat_err_components[:i+1, 0])
    line_err_max.set_data(t_i,  feat_err_components[:i+1, 1])
    line_yaw_abs.set_data(t_i,  feat_err_components[:i+1, 2])

    return (current_path_plot, camera_quiver, peg_poly_plot,
            line_err_norm, line_err_mean, line_err_max, line_yaw_abs)

ani = FuncAnimation(fig, animate, frames=len(pose_history), interval=DT*1000,
                    blit=True, repeat=False)

video_filename = 'ibvs_pentagon.mp4'
print(f"Trying to save '{video_filename}' ...")
try:
    sim_duration = len(pose_history) * DT
    fps = int(len(pose_history) / sim_duration) if sim_duration > 0 else 10
    ani.save(video_filename, writer='ffmpeg', fps=fps, dpi=100)
    print("Saved successfully.")
except Exception as ex:
    print(f"[Info] mp4 save failed (ffmpeg etc.): {ex}")

plt.show()
