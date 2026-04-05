import cv2
import csv
import time
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime


class AlertSystem:
    def __init__(self, cfg, fps):
        self.threshold = cfg["alert"]["threshold"]
        self.cooldown_frames = int(cfg["alert"]["cooldown_seconds"] * fps)
        self.pre_frames = int(cfg["alert"]["clip_pre_seconds"] * fps)
        self.post_frames = int(cfg["alert"]["clip_post_seconds"] * fps)
        self.clips_dir = Path(cfg["alert"]["clips_dir"])
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(cfg["alert"]["log_path"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        # Frame buffer for pre-event clip extraction
        self.frame_buffer = deque(maxlen=self.pre_frames)

        self.cooldown_counter = 0
        self.active_alert = False
        self.post_counter = 0
        self.clip_frames = []
        self.alert_count = 0
        self.current_event_type = "normal"

        # Init CSV log
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["alert_id", "timestamp", "frame_idx",
                                        "event_type", "anomaly_score", "clip_path"])

    def update(self, frame_bgr, frame_idx, anomaly_score, pred_class):
        """
        Feed frame into alert system. Returns alert dict if triggered, else None.
        """
        self.frame_buffer.append(frame_bgr.copy())

        alert_triggered = None

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # Collecting post-event frames
        if self.active_alert:
            self.clip_frames.append(frame_bgr.copy())
            self.post_counter -= 1
            if self.post_counter <= 0:
                clip_path = self._save_clip()
                self.active_alert = False
                alert_triggered = {
                    "clip_path": clip_path,
                    "event_type": self.current_event_type,
                }

        # New alert trigger
        if anomaly_score >= self.threshold and self.cooldown_counter == 0 and not self.active_alert:
            self.alert_count += 1
            self.current_event_type = pred_class
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            clip_name = f"alert_{self.alert_count:04d}_{pred_class}.mp4"
            clip_path = str(self.clips_dir / clip_name)

            # Start collecting clip: pre-buffer + future post frames
            self.clip_frames = list(self.frame_buffer)
            self.active_alert = True
            self.post_counter = self.post_frames
            self.cooldown_counter = self.cooldown_frames

            # Log immediately
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.alert_count, timestamp, frame_idx,
                    pred_class, f"{anomaly_score:.4f}", clip_path
                ])

            print(f"[ALERT #{self.alert_count}] frame={frame_idx} "
                  f"score={anomaly_score:.3f} type={pred_class}")

        return alert_triggered

    def _save_clip(self):
        if not self.clip_frames:
            return ""
        clip_name = f"alert_{self.alert_count:04d}_{self.current_event_type}.mp4"
        clip_path = str(self.clips_dir / clip_name)
        H, W = self.clip_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            clip_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (W, H)
        )
        for f in self.clip_frames:
            writer.write(f)
        writer.release()
        print(f"[CLIP SAVED] {clip_path}")
        self.clip_frames = []
        return clip_path