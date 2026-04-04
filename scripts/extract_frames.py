import cv2
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def extract_frames(cfg):
    st_cfg = cfg["shanghaitech"]
    frames_dir = Path(cfg["data"]["frames_dir"])
    target_fps = cfg["data"]["target_fps"]
    W = cfg["data"]["resize"]["width"]
    H = cfg["data"]["resize"]["height"]
    class_to_idx = cfg["data"]["class_to_idx"]
    win_size = cfg["sliding_window"]["window_size"]

    records = []

    # --- Training videos (all normal) ---
    train_video_dir = Path(st_cfg["training_videos"])
    train_videos = sorted(list(train_video_dir.glob("*.avi")) +
                          list(train_video_dir.glob("*.mp4")))

    for video_path in tqdm(train_videos, desc="Training (normal)"):
        clip_name = video_path.stem
        out_dir = frames_dir / "training" / clip_name
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_interval = max(1, round(src_fps / target_fps))

        frame_idx, saved = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                cv2.imwrite(str(out_dir / f"{saved:06d}.jpg"),
                            cv2.resize(frame, (W, H)))
                saved += 1
            frame_idx += 1
        cap.release()

        if saved >= win_size:
            records.append({
                "clip_name": clip_name,
                "split": "train",
                "label": "normal",
                "label_idx": class_to_idx["normal"],
                "frames_dir": str(out_dir),
                "mask_path": "",
                "num_frames": saved,
            })

    # --- Testing videos (normal + anomaly, frame-level masks) ---
    test_video_dir = Path(st_cfg["testing_videos"])
    test_mask_dir = Path(st_cfg["testing_masks"])
    test_videos = sorted(list(test_video_dir.glob("*.avi")) +
                         list(test_video_dir.glob("*.mp4")))

    for video_path in tqdm(test_videos, desc="Testing"):
        clip_name = video_path.stem
        out_dir = frames_dir / "testing" / clip_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load frame-level mask if exists
        mask_path = test_mask_dir / f"{clip_name}.npy"
        frame_labels = None
        if mask_path.exists():
            frame_labels = np.load(str(mask_path))  # binary per-frame

        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_interval = max(1, round(src_fps / target_fps))

        frame_idx, saved = 0, 0
        saved_labels = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                cv2.imwrite(str(out_dir / f"{saved:06d}.jpg"),
                            cv2.resize(frame, (W, H)))
                if frame_labels is not None and frame_idx < len(frame_labels):
                    saved_labels.append(int(frame_labels[frame_idx]))
                else:
                    saved_labels.append(0)
                saved += 1
            frame_idx += 1
        cap.release()

        if saved < win_size:
            continue

        # Save per-frame label array for this clip
        labels_arr = np.array(saved_labels, dtype=np.int32)
        labels_out = out_dir / "frame_labels.npy"
        np.save(str(labels_out), labels_arr)

        # Clip-level label: anomaly if any frame is anomalous
        has_anomaly = int(labels_arr.max())
        label_str = "anomaly" if has_anomaly else "normal"

        records.append({
            "clip_name": clip_name,
            "split": "test",
            "label": label_str,
            "label_idx": class_to_idx[label_str],
            "frames_dir": str(out_dir),
            "mask_path": str(labels_out),
            "num_frames": saved,
        })

    df = pd.DataFrame(records)
    manifest_path = Path(cfg["data"]["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"\nManifest saved: {manifest_path} — {len(df)} clips")
    print(df.groupby(["split", "label"]).size())
    return df


if __name__ == "__main__":
    cfg = load_config()
    df = extract_frames(cfg)