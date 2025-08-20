import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# Params
# ============================================================
DT = 0.1
SIM_TIME = 20.0
TOTAL_STEPS = int(SIM_TIME / DT)
FOCAL_LENGTH = 500.0
LAMBDA_MAX, LAMBDA_MIN = 2.0, 0.5
K_YAW = 1.0  # (34.27)의 -λ θ u 를 yaw-only 근사 → ωz = -K_YAW * yaw_err

# ============================================================
# Helpers (rotations)
# ============================================================
def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def Rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def Rz2(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s],
                     [s,  c]])

# ============================================================
# Geometry: tilted N-gon on a plane in world
# ============================================================
def make_ngon_world(center_w, radius=0.5, tilt_rx_deg=20.0, tilt_ry_deg=5.0, n_sides=5):
    rx = np.deg2rad(tilt_rx_deg); ry = np.deg2rad(tilt_ry_deg)
    c, s = np.cos, np.sin
    Rx = np.array([[1,0,0],[0,c(rx),-s(rx)],[0,s(rx),c(rx)]])
    Ry = np.array([[c(ry),0,s(ry)],[0,1,0],[-s(ry),0,c(ry)]])
    R_plane = Ry @ Rx
    t = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    poly_local = np.stack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)], axis=1)
    return (R_plane @ poly_local.T).T + center_w  # (N,3)

# ============================================================
# Camera model & features
# ============================================================
def world_to_cam(pts_w: np.ndarray, pose_xyzyaw: np.ndarray) -> np.ndarray:
    x, y, z, th = pose_xyzyaw
    Rcw = Rz(th).T
    tcw = -Rcw @ np.array([x, y, z])
    return (Rcw @ pts_w.T).T + tcw  # (N,3)

def project_points_to_image(pts_w, pose_xyzyaw, fpx):
    pts_c = world_to_cam(pts_w, pose_xyzyaw)
    X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
    uv = np.stack([fpx * (X / Z), fpx * (Y / Z)], axis=1)
    return uv, Z

def features_from_polygon(uv):
    # s = [u1,v1, ..., uN,vN]^T
    return uv.reshape(-1)

def get_features(pose_xyzyaw, poly_w, fpx):
    uv, Z = project_points_to_image(poly_w, pose_xyzyaw, fpx)
    s = features_from_polygon(uv)
    return s, uv, Z

# ============================================================
# Pose integration (apply camera-frame twist)
# ============================================================
def apply_cam_twist_4d(pose_xyzyaw: np.ndarray, d_cam: np.ndarray) -> np.ndarray:
    """
    pose = [x,y,z,yaw]  (world frame)
    d_cam = [dx,dy,dz,d_yaw]  (camera frame incremental motion)
    """
    x, y, z, th = pose_xyzyaw
    dx_c, dy_c, dz_c, dth = d_cam
    R_wc = Rz(th)  # camera→world
    d_world = R_wc @ np.array([dx_c, dy_c, dz_c])
    return np.array([x + d_world[0], y + d_world[1], z + d_world[2], wrap_angle(th + dth)])

# ============================================================
# Numeric interaction matrix split: L_v (vx,vy,vz), L_w (wz)
# ============================================================
def numeric_L_split_v_and_wz(pose_xyzyaw, poly_w, fpx, eps=1e-4):
    s0, _, _ = get_features(pose_xyzyaw, poly_w, fpx)
    dim = s0.size
    Lv = np.zeros((dim, 3))   # for (vx, vy, vz)
    Lw = np.zeros((dim, 1))   # for (wz) only

    # translations in cam frame
    for j, d in enumerate([np.array([eps,0,0,0]), np.array([0,eps,0,0]), np.array([0,0,eps,0])]):
        sp, _, _ = get_features(apply_cam_twist_4d(pose_xyzyaw, d), poly_w, fpx)
        Lv[:, j] = (sp - s0) / eps

    # yaw rotation only
    sp, _, _ = get_features(apply_cam_twist_4d(pose_xyzyaw, np.array([0,0,0,eps])), poly_w, fpx)
    Lw[:, 0] = (sp - s0) / eps

    return Lv, Lw

# ============================================================
# Hybrid VS control (Eq. 34.27 & 34.28)
# ============================================================
def hybrid_vs_step(pose, poly_w, s_star, fpx, lam, k_yaw):
    """
    Implements:
      ω_c = -k_yaw * yaw_err  (PBVS-like, Eq. 34.27 yaw-only)
      v_c = -L_v^+ ( λ e_t + L_ω ω_c )  (Eq. 34.28)

    Returns: new_pose, v_c (3,), omega_c (scalar), diagnostics
    """
    s, uv, Z = get_features(pose, poly_w, fpx)
    e_t = s - s_star

    # yaw-only rotation error (desired yaw = 0)
    yaw_err = wrap_angle(pose[3] - 0.0)
    omega_c = -k_yaw * yaw_err  # scalar = wz

    # L split (vx,vy,vz | wz)
    Lv, Lw = numeric_L_split_v_and_wz(pose, poly_w, fpx)

    # modified error (34.28)
    tilde_e = lam * e_t + (Lw @ np.array([omega_c]))

    # translational control (least squares)
    v_c = -np.linalg.pinv(Lv) @ tilde_e  # (3,)

    # integrate pose
    d_cam = np.array([v_c[0] * DT, v_c[1] * DT, v_c[2] * DT, omega_c * DT])
    new_pose = apply_cam_twist_4d(pose, d_cam)

    # diagnostics
    per_pt = np.linalg.norm(uv - s_star.reshape(-1,2), axis=1)
    diag = {
        "e_norm": float(np.linalg.norm(e_t)),
        "mean_duv": float(np.mean(per_pt)),
        "max_duv": float(np.max(per_pt)),
        "yaw_err_abs": float(abs(yaw_err)),
        "v_c": v_c.copy(),
        "omega_c": float(omega_c)
    }
    return new_pose, v_c, omega_c, diag

# ============================================================
# Scenario setup
# ============================================================
def setup_scene(n_sides=5):
    rim_center_world = np.array([0.0, 0.0, 2.5])
    poly_world = make_ngon_world(rim_center_world, radius=0.5,
                                 tilt_rx_deg=20.0, tilt_ry_deg=5.0,
                                 n_sides=n_sides)
    desired_pose = np.array([0.0, 0.0, 0.8, 0.0])  # [x,y,z,yaw]
    s_star, uv_star, _ = get_features(desired_pose, poly_world, FOCAL_LENGTH)

    # random initial pose (ensure front-facing)
    rng = np.random.default_rng()
    camera_pose0 = np.array([
        float(rng.uniform(-2.0, 2.0)),   # x
        float(rng.uniform(-2.0, 2.0)),   # y
        float(rng.uniform(0.5, 3.0)),    # z (positive)
        float(rng.uniform(-np.pi, np.pi))# yaw
    ])
    return rim_center_world, poly_world, desired_pose, s_star, uv_star, camera_pose0

# ============================================================
# Simulation loop
# ============================================================
def simulate():
    rim_center_world, poly_world, desired_pose, s_star, uv_star, pose = setup_scene()

    pose_hist = [pose.copy()]
    err_hist, mean_hist, max_hist, yaw_hist = [], [], [], []

    for step in range(TOTAL_STEPS):
        # lambda scheduling (heuristic but common practice)
        # combine feature error & yaw error for gain scheduling only
        s_now, _, _ = get_features(pose, poly_world, FOCAL_LENGTH)
        yaw_err = wrap_angle(pose[3])
        err_mag = np.linalg.norm(s_now - s_star) + abs(yaw_err)
        lam = max(LAMBDA_MIN, min(LAMBDA_MAX, LAMBDA_MAX * (err_mag / 50.0)))

        # hybrid VS step (Eq. 34.27 & 34.28)
        pose, v_c, omega_c, d = hybrid_vs_step(
            pose, poly_world, s_star, FOCAL_LENGTH, lam, K_YAW
        )
        pose_hist.append(pose.copy())
        err_hist.append(d["e_norm"])
        mean_hist.append(d["mean_duv"])
        max_hist.append(d["max_duv"])
        yaw_hist.append(np.rad2deg(d["yaw_err_abs"]))

        if d["e_norm"] + d["yaw_err_abs"] < 1.0:
            print(f"Goal reached! t = {step*DT:.2f} s")
            break

    return (
        np.array(pose_hist),
        np.array(err_hist),
        np.array(mean_hist),
        np.array(max_hist),
        np.array(yaw_hist),
        rim_center_world,
        poly_world
    )

# ============================================================
# Visualization
# ============================================================
def visualize(pose_hist, err_hist, mean_hist, max_hist, yaw_hist,
              rim_center_world, poly_world, video_filename="ibvs_pentagon_hybrid.mp4"):
    times = np.arange(len(err_hist)) * DT

    fig, (ax1, ax3) = plt.subplots(
        1, 2, figsize=(20, 10), dpi=150,
        gridspec_kw={'width_ratios':[1,1]}, constrained_layout=True
    )

    # --- World View (Top-down) ---
    ax1.set_title("World View (Top-down)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True); ax1.set_aspect('equal')

    # draw target polygon footprint (schematic)
    peg_radius = 0.5
    theta_draw = np.linspace(0, 2*np.pi, 64)
    ax1.plot(rim_center_world[0] + peg_radius*np.cos(theta_draw),
             rim_center_world[1] + peg_radius*np.sin(theta_draw),
             'g-', lw=2, label="Target Polygon (schematic)")
    ax1.plot(rim_center_world[0], rim_center_world[1], 'g*', ms=15, label="Target Center")

    # bounds
    x_min, x_max = np.min(pose_hist[:,0]), np.max(pose_hist[:,0])
    y_min, y_max = np.min(pose_hist[:,1]), np.max(pose_hist[:,1])
    margin = 1 * max(x_max - x_min + 1e-6, y_max - y_min + 1e-6)
    ax1.set_xlim(x_min - margin, x_max + margin)
    ax1.set_ylim(y_min - margin, y_max + margin)

    # path handles
    ax1.plot(pose_hist[:,0], pose_hist[:,1], 'b--', lw=1, alpha=0.4, label="Full Path")
    current_path_plot, = ax1.plot([], [], 'b-', lw=2, label="Robot Path", animated=True)

    camera_quiver = ax1.quiver([0],[0],[0],[0],
                               angles='xy', scale_units='xy', scale=1.0,
                               color='r', label="Camera Pose", animated=True)
    ax1.legend()

    # --- Loss Plot ---
    ax3.set_title("Feature Error & Yaw")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Error")
    ax3.grid(True)
    line_err_norm, = ax3.plot(times, err_hist, label="||e_t||")
    line_err_mean, = ax3.plot(times, mean_hist, label="Mean |Δuv| [px]")
    line_err_max,  = ax3.plot(times, max_hist,  label="Max |Δuv| [px]")
    line_yaw_abs,  = ax3.plot(times, yaw_hist,  label="|yaw_err| [deg]")
    ax3.legend()
    ax3.set_ylim(0, max(1.0, 1.1*max(
        np.max(err_hist) if len(err_hist)>0 else 1,
        np.max(mean_hist) if len(mean_hist)>0 else 1,
        np.max(max_hist)  if len(max_hist)>0  else 1,
        np.max(yaw_hist)  if len(yaw_hist)>0  else 1
    )))

    # for drawing a small "gripper polygon" oriented by yaw (2D)
    peg_local = np.stack([peg_radius*np.cos(theta_draw),
                          peg_radius*np.sin(theta_draw)], axis=1)
    peg_poly_plot, = ax1.plot([], [], 'm-', lw=2, label="Gripper Polygon", animated=True)

    def animate(i):
        current_path_plot.set_data(pose_hist[:i+1,0], pose_hist[:i+1,1])

        x, y, th = pose_hist[i,0], pose_hist[i,1], pose_hist[i,3]
        camera_quiver.set_offsets([x, y])
        camera_quiver.set_UVC(0.4*np.cos(th), 0.4*np.sin(th))

        R2 = Rz2(th)
        peg_world = peg_local @ R2.T + np.array([x, y])
        peg_poly_plot.set_data(peg_world[:,0], peg_world[:,1])

        t_i = times[:i+1]
        line_err_norm.set_data(t_i, err_hist[:i+1])
        line_err_mean.set_data(t_i, mean_hist[:i+1])
        line_err_max.set_data(t_i,  max_hist[:i+1])
        line_yaw_abs.set_data(t_i,  yaw_hist[:i+1])

        return (current_path_plot, camera_quiver, peg_poly_plot,
                line_err_norm, line_err_mean, line_err_max, line_yaw_abs)

    ani = FuncAnimation(fig, animate, frames=len(pose_hist), interval=DT*1000,
                        blit=True, repeat=False)

    print(f"Trying to save '{video_filename}' ...")
    try:
        sim_duration = max(1e-6, len(pose_hist) * DT)
        fps = int(len(pose_hist) / sim_duration)
        ani.save(video_filename, writer='ffmpeg', fps=fps, dpi=100)
        print("Saved successfully.")
    except Exception as ex:
        print(f"[Info] mp4 save failed (ffmpeg etc.): {ex}")

    plt.show()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    pose_hist, err_hist, mean_hist, max_hist, yaw_hist, rim_center_world, poly_world = simulate()
    visualize(pose_hist, err_hist, mean_hist, max_hist, yaw_hist,
              rim_center_world, poly_world)
