import argparse
import os
import re
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # use Agg canvas so tostring_rgb works everywhere
import matplotlib.pyplot as plt

try:
    import cv2  # only needed for mp4 writing (optional)
except Exception:
    cv2 = None


def natural_frame_id(s):
    # Extract zero-padded numeric frame id like 0001 from strings like frame_0001
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else None


def load_xyz_point(xyz_dir, frame_id_str, u, v, patch=3):
    """
    Load a small patch around (u,v) from xyz/frame_XXXX.npz and return
    the averaged (X,Y,Z) in camera coordinates (meters).
    - patch: odd integer window size (e.g., 1 uses exactly the center pixel).
    """
    path = os.path.join(xyz_dir, f"frame_{frame_id_str}.npz")
    arr = np.load(path)["points"]  # (H, W, 3)

    h, w, _ = arr.shape
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))

    if patch <= 1:
        xyz = arr[v, u].astype(np.float64)
        if not np.all(np.isfinite(xyz)):
            return None
        return xyz

    r = patch // 2
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)

    patch_vals = arr[v0:v1, u0:u1].reshape(-1, 3).astype(np.float64)
    good = np.all(np.isfinite(patch_vals), axis=1)
    if not np.any(good):
        return None

    return patch_vals[good].mean(axis=0)


def main():
    ap = argparse.ArgumentParser(description="Ego-trajectory from traffic-light depth")
    ap.add_argument("--csv", required=True,
                    help="CSV with columns: frame_id,x_min,y_min,x_max,y_max")
    ap.add_argument("--xyz_dir", default="xyz",
                    help="Directory containing frame_XXXX.npz files with key 'points'")
    ap.add_argument("--patch", type=int, default=3,
                    help="Odd window size to average around (u,v). Use 1 to sample the single pixel.")
    ap.add_argument("--video", action="store_true",
                    help="If set, writes trajectory.mp4 (requires OpenCV).")
    ap.add_argument("--fps", type=int, default=20, help="FPS for mp4 if --video set.")
    ap.add_argument("--out", default="trajectory.png", help="Output PNG filename.")
    args = ap.parse_args()

    if args.patch % 2 == 0 or args.patch < 1:
        raise ValueError("--patch must be an odd integer >= 1")

    df = pd.read_csv(args.csv)
    # Ensure expected columns
    for col in ["frame_id", "x_min", "y_min", "x_max", "y_max"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV.")

    # Sort by numeric frame index if possible
    df["_fid"] = df["frame_id"].apply(natural_frame_id)
    df = df.sort_values(by="_fid").reset_index(drop=True)

    traj = []  # list of (x_world, y_world)
    frame_ids_str = []

    for _, row in df.iterrows():
        fid = row["frame_id"]
        fid_str = f"{natural_frame_id(fid):04d}" if pd.notnull(fid) else None
        if fid_str is None:
            continue

        # center (u,v) in pixel coords
        u = 0.5 * (row["x_min"] + row["x_max"])
        v = 0.5 * (row["y_min"] + row["y_max"])

        xyz = load_xyz_point(args.xyz_dir, fid_str, u, v, patch=args.patch)
        if xyz is None:
            # skip this frame if depth invalid
            continue

        X_cam, Y_cam, Z_cam = xyz  # meters, camera coords (+X fwd, +Y right, +Z up)

        # Map to world BEV where +X forward, +Y left, origin under light
        # Car position relative to light column on ground (ignore height):
        x_world = -X_cam
        y_world = -Y_cam
        traj.append((x_world, y_world))
        frame_ids_str.append(fid_str)

    if not traj:
        raise RuntimeError("No valid XYZ points found from the provided CSV/NPZ data.")

    traj = np.array(traj)  # (N,2)
    xs, ys = traj[:, 0], traj[:, 1]

    # --- Plot static trajectory ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)  [forward]")
    ax.set_ylabel("Y (m)  [left]")
    ax.set_title("Ego-vehicle Trajectory (BEV)")
    # Put the traffic light at (0,0) as reference marker:
    ax.scatter([0], [0], marker="*", s=120, label="Traffic Light")
    ax.legend(loc="best")
    # Nice margins
    pad = 2.0
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    # --- Optional video ---
    if args.video:
        if cv2 is None:
            print("OpenCV not installed; cannot write mp4. Install opencv-python.")
            return

        import io

        tmp_w, tmp_h = 720, 720
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter("trajectory.mp4", fourcc, args.fps, (tmp_w, tmp_h))

        pad = 2.0
        x_min, x_max = xs.min() - pad, xs.max() + pad
        y_min, y_max = ys.min() - pad, ys.max() + pad

        for i in range(1, len(xs) + 1):
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.scatter(xs[:i], ys[:i], s=12)
            ax2.scatter([0], [0], marker="*", s=120, label="Traffic Light")
            ax2.set_aspect("equal", adjustable="box")
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(y_min, y_max)
            ax2.set_xlabel("X (m)  [forward]")
            ax2.set_ylabel("Y (m)  [left]")
            ax2.set_title(f"Ego Trajectory (t = {i-1})")
            ax2.legend(loc="best")
            fig2.tight_layout()

            # Backend-agnostic: render to PNG in memory, then decode with OpenCV
            buf = io.BytesIO()
            fig2.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            png_bytes = np.frombuffer(buf.read(), dtype=np.uint8)
            frame = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)  # BGR
            buf.close()
            plt.close(fig2)

            frame = cv2.resize(frame, (tmp_w, tmp_h), interpolation=cv2.INTER_AREA)
            vw.write(frame)

        vw.release()
        print("Wrote trajectory.mp4")



if __name__ == "__main__":
    main()
