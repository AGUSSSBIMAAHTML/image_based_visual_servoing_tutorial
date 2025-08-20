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

def Rz2(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s],
                     [s,  c]])

# --- Geometry: tilted plane N-gon in world ---
def make_ngon_world(center_w, radius=0.5, tilt_rx_deg=20.0, tilt_ry_deg=5.0, n_sides=5):
    """Return N polygon vertices on a tilted plane around center_w."""
    rx = np.deg2rad(tilt_rx_deg); ry = np.deg2rad(tilt_ry_deg)
    c,s = np.cos, np.sin
    Rx = np.array([[1,0,0],[0,c(rx),-s(rx)],[0,s(rx),c(rx)]])
    Ry = np.array([[c(ry),0,s(ry)],[0,1,0],[-s(ry),0,c(ry)]])
    R_plane = Ry @ Rx
    t = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    poly_local = np.stack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)], axis=1)
    return (R_plane @ poly_local.T).T + center_w  # (N,3)

def project_points_to_image(pts_w, pose_xyth, fpx):
    x, y, th = pose_xyth
    Rcw = Rz(th).T
    tcw = -Rcw @ np.array([x, y, 0.0])
    pts_c = (Rcw @ pts_w.T).T + tcw
    X, Y, Z = pts_c[:,0], pts_c[:,1], pts_c[:,2]
    uv = np.stack([fpx*(X/Z), fpx*(Y/Z)], axis=1)
    return uv, Z

# --- Features (pentagon: 5 vertices -> 10D feature) ---
def features_from_polygon(uv):
    return uv.reshape(-1)

def get_features(pose_xyth, poly_w, fpx):
    uv, _ = project_points_to_image(poly_w, pose_xyth, fpx)
    s = features_from_polygon(uv)
    return s, uv

def feature_error(s, s_star):
    return s - s_star

# --- Pose update helper (camera twist in camera frame) ---
def apply_cam_twist_to_pose(pose_xyth, d_cam):
    x, y, th = pose_xyth
    dx_c, dy_c, dth = d_cam
    c, s = np.cos(th), np.sin(th)
    dx_w = c*dx_c - s*dy_c
    dy_w = s*dx_c + c*dy_c
    return np.array([x + dx_w, y + dy_w, wrap_angle(th + dth)])

# --- Numeric interaction matrix for translation (2 DOF: x/y in cam frame) ---
def numeric_interaction_matrix_3dof(pose_xyth, poly_w, fpx, eps_t=1e-4):
    s0, _ = get_features(pose_xyth, poly_w, fpx)     # dim = 2*N
    L = np.zeros((s0.size, 2))                       # yaw is controlled separately
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
    s, uv = get_features(camera_pose, poly_world, FOCAL_LENGTH)

    e = feature_error(s, desired_features)  # (10,)
    yaw_err = wrap_angle(camera_pose[2] - desired_pose[2])

    # per-vertex pixel error magnitudes
    duv = (uv - desired_uv)                  # (5,2)
    per_pt = np.linalg.norm(duv, axis=1)     # (5,)
    mean_abs = float(np.mean(per_pt))
    max_abs  = float(np.max(per_pt))

    error_norm = np.linalg.norm(e) + abs(yaw_err)

    error_history.append(error_norm)
    feat_err_components.append([mean_abs, max_abs, abs(yaw_err)])

    current_lambda = max(LAMBDA_MIN, min(LAMBDA_MAX, LAMBDA_MAX*(error_norm/50.0)))

    if error_norm < 1.0:
        print(f"Goal reached! t = {step*DT:.2f} s")
        break

    # translation IBVS (x,y in cam frame)
    L = numeric_interaction_matrix_3dof(camera_pose, poly_world, FOCAL_LENGTH)
    v_trans = -current_lambda * (np.linalg.pinv(L) @ e)   # (2,)

    # yaw control (separate P control)
    v_yaw = -K_YAW * yaw_err

    # update pose
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

# 범위 계산: 카메라 경로 + 타겟 폴리곤 모두 포함
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
feat_err_components[:,2]= np.rad2deg(feat_err_components[:,2])  # yaw error in degrees
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
        np.max(feat_err_components[:,2])  # ← deg 기준
    ]) * 1.1
))

def animate(i):
    # World view path
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
