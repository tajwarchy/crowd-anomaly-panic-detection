import cv2
import yaml
import numpy as np
import torch
import collections
from pathlib import Path
from src.density import DensityStream
from src.flow import FlowExtractor
from src.model import load_model
from src.utils import draw_flow_vectors, draw_density_overlay


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class InferencePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.win_size = cfg["sliding_window"]["window_size"]
        self.feat_h = cfg["sliding_window"]["feature_height"]
        self.feat_w = cfg["sliding_window"]["feature_width"]
        self.frame_skip = cfg["inference"]["frame_skip"]
        self.threshold = cfg["alert"]["threshold"]
        self.classes = cfg["data"]["classes"]

        # Streams
        self.density_stream = DensityStream(cfg)
        self.flow_extractor = FlowExtractor(cfg)
        self.model = load_model(cfg)

        # Rolling window buffer: deque of (density_map, flow_map) pairs
        self.buffer = collections.deque(maxlen=self.win_size)
        self.prev_frame = None
        self.frame_count = 0
        self.score_history = []   # anomaly score per frame
        self.class_history = []   # predicted class per frame

    def reset(self):
        self.buffer.clear()
        self.prev_frame = None
        self.frame_count = 0
        self.score_history = []
        self.class_history = []

    def _fast_density(self, frame_bgr):
        """Proxy density until V3.1 weights are available."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32)
        blurred = cv2.GaussianBlur(edges, (21, 21), 0)
        resized = cv2.resize(blurred, (self.feat_w, self.feat_h))
        max_val = resized.max()
        if max_val > 0:
            resized /= max_val
        return resized.astype(np.float32)

    def process_frame(self, frame_bgr):
        """
        Process one frame. Returns dict with:
          - anomaly_score: float [0,1]
          - pred_class: str
          - density_map: np.ndarray (feat_h, feat_w)
          - flow_map: np.ndarray (feat_h, feat_w)
          - flow_raw: np.ndarray (H, W, 2) or None
          - is_anomaly: bool
        """
        self.frame_count += 1

        # Density map
        dmap = self._fast_density(frame_bgr)

        # Flow map
        if self.prev_frame is None:
            fmap = np.zeros((self.feat_h, self.feat_w), dtype=np.float32)
            flow_raw = None
        else:
            result = self.flow_extractor.compute(self.prev_frame, frame_bgr)
            fmap = result["magnitude_map"]
            flow_raw = result["flow"]

        self.prev_frame = frame_bgr.copy()
        self.buffer.append(np.stack([dmap, fmap], axis=0))

        # Run classifier every frame_skip frames if buffer is full
        anomaly_score = self.score_history[-1] if self.score_history else 0.0
        pred_class = self.class_history[-1] if self.class_history else "normal"

        if len(self.buffer) == self.win_size and self.frame_count % self.frame_skip == 0:
            window = np.stack(list(self.buffer), axis=0)  # (T, 2, H, W)
            tensor = torch.from_numpy(window).float().unsqueeze(0)  # (1, T, 2, H, W)
            with torch.no_grad():
                out = self.model(tensor)
                probs = torch.softmax(out, dim=1).squeeze().numpy()
            anomaly_score = float(probs[1])
            pred_class = self.classes[int(probs.argmax())]

        self.score_history.append(anomaly_score)
        self.class_history.append(pred_class)

        return {
            "anomaly_score": anomaly_score,
            "pred_class": pred_class,
            "density_map": dmap,
            "flow_map": fmap,
            "flow_raw": flow_raw,
            "is_anomaly": anomaly_score >= self.threshold,
        }