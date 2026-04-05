import cv2
import numpy as np
import torch
import yaml
from pathlib import Path


class FarnebackFlow:
    def __init__(self, cfg):
        fb = cfg["flow"]["farneback"]
        self.pyr_scale = fb["pyr_scale"]
        self.levels = fb["levels"]
        self.winsize = fb["winsize"]
        self.iterations = fb["iterations"]
        self.poly_n = fb["poly_n"]
        self.poly_sigma = fb["poly_sigma"]
        self.flags = fb["flags"]
        self.out_h = cfg["sliding_window"]["feature_height"]
        self.out_w = cfg["sliding_window"]["feature_width"]
        grid_cfg = cfg["flow"]["entropy_grid"]
        self.grid_rows = grid_cfg["rows"]
        self.grid_cols = grid_cfg["cols"]

    def compute(self, frame1_bgr: np.ndarray, frame2_bgr: np.ndarray) -> dict:
        gray1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma, self.flags
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalize and resize magnitude map to feature size
        mag_norm = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        mag_resized = cv2.resize(mag_norm, (self.out_w, self.out_h))

        # Directional entropy per grid region
        entropy = self._directional_entropy(angle, magnitude)

        return {
            "flow": flow,
            "magnitude": magnitude,
            "angle": angle,
            "magnitude_map": mag_resized.astype(np.float32),
            "directional_entropy": entropy,
        }

    def _directional_entropy(self, angle: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
        H, W = angle.shape
        cell_h = H // self.grid_rows
        cell_w = W // self.grid_cols
        entropy_map = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cell_angle = angle[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                cell_mag = magnitude[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]

                # Weight angles by magnitude, bin into 8 directions
                weights = cell_mag.flatten()
                angles = cell_angle.flatten()
                if weights.sum() < 1e-6:
                    entropy_map[r, c] = 0.0
                    continue

                bins, _ = np.histogram(angles, bins=8, range=(0, 2*np.pi), weights=weights)
                bins = bins / (bins.sum() + 1e-8)
                # Shannon entropy
                entropy_map[r, c] = -np.sum(bins * np.log(bins + 1e-8))

        return entropy_map


class FlowExtractor:
    def __init__(self, cfg):
        self.method = cfg["flow"]["method"]
        self.cfg = cfg
        if self.method == "farneback":
            self.extractor = FarnebackFlow(cfg)
        elif self.method == "raft":
            self.extractor = RAFTFlow(cfg)
        else:
            raise ValueError(f"Unknown flow method: {self.method}")

    def compute(self, frame1_bgr: np.ndarray, frame2_bgr: np.ndarray) -> dict:
        return self.extractor.compute(frame1_bgr, frame2_bgr)


class RAFTFlow:
    """RAFT optical flow — MPS with CPU fallback."""
    def __init__(self, cfg):
        self.out_h = cfg["sliding_window"]["feature_height"]
        self.out_w = cfg["sliding_window"]["feature_width"]
        grid_cfg = cfg["flow"]["entropy_grid"]
        self.grid_rows = grid_cfg["rows"]
        self.grid_cols = grid_cfg["cols"]

        try:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            weights = Raft_Small_Weights.DEFAULT
            self.transforms = weights.transforms()
            self.model = raft_small(weights=weights)
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self.model = self.model.to(self.device).eval()
            print(f"[RAFTFlow] Loaded on {self.device}")
        except Exception as e:
            print(f"[RAFTFlow] Failed to load: {e} — falling back to Farneback")
            self.model = None

        # Farneback fallback
        self._fb = FarnebackFlow(cfg)

    def compute(self, frame1_bgr: np.ndarray, frame2_bgr: np.ndarray) -> dict:
        if self.model is None:
            return self._fb.compute(frame1_bgr, frame2_bgr)

        try:
            import torchvision.transforms.functional as F_tv

            def to_tensor(bgr):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
                return t

            t1 = to_tensor(frame1_bgr)
            t2 = to_tensor(frame2_bgr)
            t1, t2 = self.transforms(t1, t2)
            t1, t2 = t1.to(self.device), t2.to(self.device)

            with torch.no_grad():
                flow_preds = self.model(t1, t2)
            flow = flow_preds[-1].squeeze().cpu().numpy().transpose(1, 2, 0)

            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag_norm = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
            mag_resized = cv2.resize(mag_norm, (self.out_w, self.out_h))
            entropy = self._fb._directional_entropy(angle, magnitude)

            return {
                "flow": flow,
                "magnitude": magnitude,
                "angle": angle,
                "magnitude_map": mag_resized.astype(np.float32),
                "directional_entropy": entropy,
            }
        except Exception as e:
            print(f"[RAFTFlow] Inference failed: {e} — falling back to Farneback")
            return self._fb.compute(frame1_bgr, frame2_bgr)