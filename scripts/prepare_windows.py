import cv2
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.flow import FlowExtractor


def fast_density_proxy(frame_bgr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32)
    blurred = cv2.GaussianBlur(edges, (21, 21), 0)
    resized = cv2.resize(blurred, (out_w, out_h))
    max_val = resized.max()
    if max_val > 0:
        resized /= max_val
    return resized.astype(np.float32)

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_windows(cfg):
    manifest = pd.read_csv(cfg["data"]["manifest_path"])
    windows_dir = Path(cfg["data"]["windows_dir"])
    windows_dir.mkdir(parents=True, exist_ok=True)

    win_size = cfg["sliding_window"]["window_size"]
    stride = cfg["sliding_window"]["stride"]
    feat_h = cfg["sliding_window"]["feature_height"]
    feat_w = cfg["sliding_window"]["feature_width"]

    flow_extractor = FlowExtractor(cfg)

    records = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Generating windows"):
        clip_dir = Path(row["frames_dir"])
        frame_files = sorted(clip_dir.glob("*.jpg"))
        if len(frame_files) < win_size:
            continue

        # Load per-frame labels if available
        mask_path = clip_dir / "frame_labels.npy"
        frame_labels = np.load(str(mask_path)) if mask_path.exists() else \
                       np.zeros(len(frame_files), dtype=np.int32)

        # Read all frames for this clip
        # Subsample frames to reduce load — keep enough for all windows
        max_frames = min(len(frame_files), (win_size * 4))
        step = max(1, len(frame_files) // max_frames)
        frame_files_sub = frame_files[::step]
        frame_labels = frame_labels[::step] if len(frame_labels) > 0 else frame_labels

        frames = []
        for fp in frame_files_sub:
            f = cv2.imread(str(fp))
            if f is not None:
                frames.append(f)

        if len(frames) < win_size:
            continue

        # Precompute density maps and flow magnitude maps
        density_maps = []
        flow_maps = []

        for i, frame in enumerate(frames):
            dmap = fast_density_proxy(frame, feat_h, feat_w)
            density_maps.append(dmap)

            if i == 0:
                # No previous frame for first — use zeros
                fmap = np.zeros((feat_h, feat_w), dtype=np.float32)
            else:
                result = flow_extractor.compute(frames[i-1], frame)
                fmap = result["magnitude_map"]
            flow_maps.append(fmap)

        # Slide window
        n_frames = len(frames)
        win_idx = 0
        for start in range(0, n_frames - win_size + 1, stride):
            end = start + win_size

            # Stack: shape (win_size, 2, feat_h, feat_w)
            window = np.stack([
                np.stack([density_maps[i], flow_maps[i]], axis=0)
                for i in range(start, end)
            ], axis=0).astype(np.float32)

            # Window label: majority vote of frame labels in window
            win_frame_labels = frame_labels[start:end] if end <= len(frame_labels) \
                               else np.zeros(win_size, dtype=np.int32)
            win_label = int(np.round(win_frame_labels.mean()))

            # Save window
            clip_name = row["clip_name"]
            split = row["split"]
            fname = f"{clip_name}_w{win_idx:04d}.npy"
            out_path = windows_dir / split / fname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), window)

            records.append({
                "window_path": str(out_path),
                "clip_name": clip_name,
                "split": split,
                "start_frame": start,
                "end_frame": end,
                "label": win_label,
            })
            win_idx += 1

    df = pd.DataFrame(records)
    out_csv = Path(cfg["data"]["windows_dir"]) / "windows_manifest.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWindows manifest saved: {out_csv}")
    print(f"Total windows: {len(df)}")
    print(df.groupby(["split", "label"]).size())
    return df


if __name__ == "__main__":
    cfg = load_config()
    prepare_windows(cfg)