# scripts/burn_labels.py
import cv2
import yaml
import pandas as pd
from pathlib import Path


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        import yaml
        return yaml.safe_load(f)


def burn_labels(cfg):
    log_path = Path(cfg["alert"]["log_path"])
    if not log_path.exists():
        print("No alerts log found — run inference on demo video first")
        return

    df = pd.read_csv(log_path)
    if df.empty:
        print("No alerts logged")
        return

    for _, row in df.iterrows():
        clip_path = Path(row["clip_path"])
        if not clip_path.exists():
            continue

        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = clip_path.parent / f"{clip_path.stem}_labeled.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (W, H)
        )

        event_type = str(row["event_type"]).upper()
        score = float(row["anomaly_score"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Red border
            cv2.rectangle(frame, (0, 0), (W-1, H-1), (0, 0, 255), 4)
            # Event label background
            cv2.rectangle(frame, (0, H-50), (W, H), (0, 0, 0), -1)
            cv2.putText(frame, f"EVENT: {event_type}  Score: {score:.3f}",
                        (10, H-15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            writer.write(frame)

        cap.release()
        writer.release()
        print(f"Labeled clip saved: {out_path}")


if __name__ == "__main__":
    cfg = load_config()
    burn_labels(cfg)