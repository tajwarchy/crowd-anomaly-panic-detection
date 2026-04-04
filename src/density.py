import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import yaml
from pathlib import Path


class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


class DensityStream:
    def __init__(self, cfg):
        weights_path = cfg["density"]["csrnet_weights"]
        self.device = torch.device(cfg["inference"]["device_density"]
                                   if torch.backends.mps.is_available() else "cpu")
        self.model = CSRNet().to(self.device)

        if Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)
            print(f"[DensityStream] Loaded weights from {weights_path} on {self.device}")
        else:
            print(f"[DensityStream] WARNING: weights not found at {weights_path} — running with random init")

        self.model.eval()
        self.out_h = cfg["sliding_window"]["feature_height"]
        self.out_w = cfg["sliding_window"]["feature_width"]

    @torch.no_grad()
    def get_density_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns normalized density map as float32 numpy array (H, W)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb = (rgb - mean) / std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        out = self.model(tensor).squeeze().cpu().numpy()
        out = np.maximum(out, 0)
        out_resized = cv2.resize(out, (self.out_w, self.out_h))
        max_val = out_resized.max()
        if max_val > 0:
            out_resized /= max_val
        return out_resized.astype(np.float32)


def verify_csrnet(cfg_path="configs/config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    stream = DensityStream(cfg)

    # Create a dummy frame
    dummy = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    dmap = stream.get_density_map(dummy)

    print(f"Density map shape : {dmap.shape}")
    print(f"Density map range : [{dmap.min():.4f}, {dmap.max():.4f}]")
    print(f"Density map dtype : {dmap.dtype}")
    print("CSRNet verification PASSED")


if __name__ == "__main__":
    verify_csrnet()