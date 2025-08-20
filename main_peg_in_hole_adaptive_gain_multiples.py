import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Params ---
DT = 0.1
SIM_TIME = 20
TOTAL_STEPS = int(SIM_TIME / DT)
FOCAL_LENGTH = 500.0
LAMBDA_MAX, LAMBDA_MIN = 2.0, 0.5

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

def ellipse_from_points(uv):
    u0, v0 = uv.mean(axis=0)
    duv = uv - np.array([u0, v0])
    C = (duv.T @ duv) / max(uv.shape[0], 1)
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]; evals, evecs = evals[idx], evecs[:, idx]
    a, b = np.sqrt(np.maximum(evals, 1e-12))
    alpha = wrap_angle(np.arctan2(evecs[1,0], evecs[0,0]))
    return float(u0), float(v0), float(a), float(b), float(alpha)

def features_from_ellipse(u0,v0,a,b,alpha):
    return np.array([u0, v0, np.log(max(a*b, 1e-12)), alpha])

def get_features(pose_xyth, rim_w, fpx):
    uv,_ = project_points_to_image(rim_w, pose_xyth, fpx)
    return features_from_ellipse(*ellipse_from_points(uv)), uv

def feature_error(s, s_star):
    e = s - s_star; e[-1] = wrap_angle(e[-1]); return e

# --- Interaction matrix ---
def apply_cam_twist_to_pose(pose_xyth, d_cam):
    x,y,th = pose_xyth; dx_c,dy_c,dth = d_cam
    c,s = np.cos(th), np.sin(th)
    dx_w = c*dx_c - s*dy_c; dy_w = s*dx_c + c*dy_c
    return np.array([x+dx_w, y+dy_w, wrap_angle(th+dth)])

def numeric_interaction_matrix_3dof(pose_xyth, rim_w, fpx, eps_t=1e-4, eps_r=1e-4):
    s0,_ = get_features(pose_xyth, rim_w, fpx)
    L = np.zeros((s0.size, 3))
    deltas = [np.array([eps_t,0,0]), np.array([0,eps_t,0]), np.array([0,0,eps_r])]
    for j,d in enumerate(deltas):
        pose_p = apply_cam_twist_to_pose(pose_xyth, d)
        s1,_ = get_features(pose_p, rim_w, fpx)
        ds = s1 - s0; ds[-1] = wrap_angle(ds[-1])
        step = d[j] if j<2 else eps_r
        L[:,j] = ds/step
    return L

# --- Simulation wrapper ---
def simulate_ibvs(k: int = 5):
    rim_center_world = np.array([0,0,2.5])
    rim_world = make_rim_world(rim_center_world,0.5,20,5,200)
    desired_pose = np.array([0.0,0.0,0.0])
    desired_features,_ = get_features(desired_pose, rim_world, FOCAL_LENGTH)

    all_pose_histories, all_feature_histories = [], []
    frames_per_trial = []

    for trial in range(k):
        init_pose = np.array([
            np.random.uniform(-3,3),
            np.random.uniform(-1,4),
            np.random.uniform(-np.pi,np.pi)
        ])
        camera_pose = init_pose.copy()
        pose_history, feature_history = [camera_pose.copy()], []

        for step in range(TOTAL_STEPS):
            s, uv = get_features(camera_pose, rim_world, FOCAL_LENGTH)
            feature_history.append(uv)
            e = feature_error(s, desired_features)
            error_norm = np.linalg.norm(e)
            lam = max(LAMBDA_MIN, min(LAMBDA_MAX, LAMBDA_MAX*(error_norm/50)))
            if error_norm < 1.0: break
            L = numeric_interaction_matrix_3dof(camera_pose, rim_world, FOCAL_LENGTH)
            v_cam = -lam * (np.linalg.pinv(L) @ e)
            x,y,th = camera_pose; c,s = np.cos(th), np.sin(th)
            vx_w = c*v_cam[0]-s*v_cam[1]; vy_w = s*v_cam[0]+c*v_cam[1]; dth=v_cam[2]
            camera_pose = np.array([x+vx_w*DT, y+vy_w*DT, wrap_angle(th+dth*DT)])
            pose_history.append(camera_pose.copy())

        all_pose_histories.append(np.array(pose_history))
        all_feature_histories.append(np.array(feature_history, dtype=object))
        frames_per_trial.append(len(pose_history))

    total_frames = sum(frames_per_trial)

    # --- Visualization ---
    fig = plt.figure(figsize=(20,12), dpi=150)
    ax1 = fig.add_subplot(1,2,1); ax2 = fig.add_subplot(1,2,2)

    ax1.set_title("World View (Top-down)")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.grid(True); ax1.set_aspect('equal')

    # Axis auto scale
    all_x = np.concatenate([ph[:,0] for ph in all_pose_histories])
    all_y = np.concatenate([ph[:,1] for ph in all_pose_histories])
    mx = 0.1 * (np.ptp(all_x) if np.ptp(all_x) > 0 else 1.0)
    my = 0.1 * (np.ptp(all_y) if np.ptp(all_y) > 0 else 1.0)
    ax1.set_xlim(all_x.min()-mx, all_x.max()+mx)
    ax1.set_ylim(all_y.min()-my, all_y.max()+my)

    # Rim
    theta = np.linspace(0,2*np.pi,200)
    ax1.plot(rim_center_world[0]+0.5*np.cos(theta),
             rim_center_world[1]+0.5*np.sin(theta),'g-',lw=2,label="Rim Hole")
    ax1.legend()

    # Artists
    path_line, = ax1.plot([],[],'b-',lw=2,label="Path")
    peg_line,  = ax1.plot([],[],'m-',lw=2,label="Gripper Peg")
    # camera_arrow = ax1.quiver([],[],[],[],color='r',angles='xy',
    #                           scale_units='xy',scale=1,width=0.01,label="Camera Heading")
    camera_arrow = ax1.quiver([0], [0], [1], [0],
                            color='r', angles='xy',
                            scale_units='xy', scale=1, width=0.02)
    info_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes,
                         ha="left", va="top", fontsize=12,
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    ax2.set_title("Camera View (Ellipse Projection)")
    ax2.set_xlabel("u (px)"); ax2.set_ylabel("v (px)")
    ax2.set_aspect('equal'); ax2.grid(True)
    des_uv,_ = project_points_to_image(rim_world, desired_pose, FOCAL_LENGTH)
    ax2.plot(des_uv[:,0], des_uv[:,1],'r.',ms=2,label="Desired Rim")
    rim_plot, = ax2.plot([],[],'b.',ms=2,label="Current Rim")
    ax2.legend(); ax2.set_xlim(-200,200); ax2.set_ylim(-200,200)

    # Animate
    def animate(i):
        trial_idx, offset = 0, i
        while offset >= frames_per_trial[trial_idx]:
            offset -= frames_per_trial[trial_idx]; trial_idx += 1

        pose_history = all_pose_histories[trial_idx]
        feature_history = all_feature_histories[trial_idx]

        # World view update
        path_line.set_data(pose_history[:offset+1,0], pose_history[:offset+1,1])
        x,y,th = pose_history[offset]
        peg_line.set_data(x+0.5*np.cos(theta), y+0.5*np.sin(theta))
        camera_arrow.set_offsets([x,y])
        camera_arrow.set_UVC(0.5*np.cos(th),0.5*np.sin(th))

        # Camera view update
        uv = feature_history[offset]
        rim_plot.set_data(uv[:,0], uv[:,1])

        # Error / λ 표시
        s, _ = get_features(pose_history[offset], rim_world, FOCAL_LENGTH)
        e = feature_error(s, desired_features)
        error_norm = np.linalg.norm(e)
        lam = max(LAMBDA_MIN, min(LAMBDA_MAX, LAMBDA_MAX*(error_norm/50)))
        info_text.set_text(f"Trial {trial_idx+1}/{k}\nStep {offset}\nError: {error_norm:.2f}\nλ: {lam:.2f}")

        return path_line, peg_line, rim_plot, camera_arrow, info_text

    ani = FuncAnimation(fig, animate, frames=total_frames,
                        interval=DT*1000, blit=True, repeat=False)

    ani.save("ibvs_multi_trials.mp4", writer="ffmpeg", fps=int(1/DT), dpi=150)
    plt.show()

# 실행
simulate_ibvs(1)  # k=5가 디폴트
