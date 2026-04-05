import cv2
import yaml
import numpy as np
import argparse
from pathlib import Path
from src.pipeline import InferencePipeline
from src.alert import AlertSystem
from src.utils import draw_flow_vectors, draw_density_overlay


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def draw_timeline(frame_bgr, score_history, win_w, timeline_h=80, max_frames=300):
    """Draw anomaly score timeline bar at bottom of frame."""
    timeline = np.zeros((timeline_h, win_w, 3), dtype=np.uint8)
    timeline[:] = (20, 20, 20)

    # Threshold line
    thresh_y = int(timeline_h * 0.4)
    cv2.line(timeline, (0, thresh_y), (win_w, thresh_y), (0, 180, 255), 1)

    # Score graph
    history = score_history[-max_frames:]
    n = len(history)
    if n >= 2:
        pts = []
        for i, s in enumerate(history):
            x = int(i / max(n - 1, 1) * (win_w - 1))
            y = int((1.0 - s) * (timeline_h - 4)) + 2
            pts.append((x, y))
        for i in range(len(pts) - 1):
            color = (0, 100, 255) if history[i] >= 0.6 else (0, 220, 100)
            cv2.line(timeline, pts[i], pts[i+1], color, 2)

    # Label
    cv2.putText(timeline, "Anomaly Score", (5, timeline_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return np.vstack([frame_bgr, timeline])


def draw_hud(frame, frame_idx, anomaly_score, pred_class, is_anomaly, fps_val):
    H, W = frame.shape[:2]
    # Background bar
    cv2.rectangle(frame, (0, 0), (W, 36), (0, 0, 0), -1)

    status_color = (0, 0, 255) if is_anomaly else (0, 220, 80)
    status_text = f"[ANOMALY: {pred_class.upper()}]" if is_anomaly else "[NORMAL]"

    cv2.putText(frame, status_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Score: {anomaly_score:.3f}", (W - 200, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Frame: {frame_idx}  FPS: {fps_val:.1f}", (W // 2 - 80, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def run(video_path, cfg, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = cfg["inference"]["output_width"]
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    H = int(W * src_h / src_w)
    timeline_h = cfg["inference"]["timeline_height"]
    out_h = H + timeline_h

    # Output video
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(cfg["output"]["video_dir"]) / f"{stem}_annotated.mp4")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps,
        (W, out_h)
    )

    pipeline = InferencePipeline(cfg)
    alert_sys = AlertSystem(cfg, fps=src_fps)

    import time
    frame_idx = 0
    fps_val = 0.0
    t_start = time.perf_counter()

    print(f"Processing: {video_path}")
    print(f"Output    : {output_path}")
    print(f"Src FPS   : {src_fps:.1f}  Target W: {W}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (W, H))
        result = pipeline.process_frame(frame)

        # Dual overlay: density + flow vectors
        vis = draw_density_overlay(frame.copy(), result["density_map"], alpha=0.35)
        if result["flow_raw"] is not None:
            vis = draw_flow_vectors(vis, result["flow_raw"], step=20, scale=1.5)

        # Alert flash border
        if result["is_anomaly"]:
            cv2.rectangle(vis, (0, 0), (W-1, H-1), (0, 0, 255), 4)

        # HUD
        elapsed = time.perf_counter() - t_start
        fps_val = frame_idx / elapsed if elapsed > 0 else 0.0
        vis = draw_hud(vis, frame_idx, result["anomaly_score"],
                       result["pred_class"], result["is_anomaly"], fps_val)

        # Timeline
        vis = draw_timeline(vis, pipeline.score_history, W, timeline_h)

        writer.write(vis)

        # Alert system
        alert_sys.update(frame.copy(), frame_idx,
                         result["anomaly_score"], result["pred_class"])

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx:5d} | score={result['anomaly_score']:.3f} "
                  f"| fps={fps_val:.1f}")

    cap.release()
    writer.release()
    print(f"\nDone. {frame_idx} frames processed.")
    print(f"Annotated video : {output_path}")
    print(f"Alert log       : {cfg['alert']['log_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default=None, help="Output video path (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(args.video, cfg, args.output)