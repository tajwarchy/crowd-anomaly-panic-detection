import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_flow_vectors(frame_bgr: np.ndarray, flow: np.ndarray,
                      step: int = 16, scale: float = 1.0) -> np.ndarray:
    """Draw optical flow vectors on frame."""
    vis = frame_bgr.copy()
    H, W = flow.shape[:2]
    y_coords, x_coords = np.mgrid[step//2:H:step, step//2:W:step]

    for y, x in zip(y_coords.flatten(), x_coords.flatten()):
        fx, fy = flow[y, x]
        mag = np.sqrt(fx**2 + fy**2)
        if mag < 0.5:
            continue
        ex = int(x + fx * scale)
        ey = int(y + fy * scale)
        ex = np.clip(ex, 0, W - 1)
        ey = np.clip(ey, 0, H - 1)
        color_val = min(int(mag * 10), 255)
        color = (0, 255 - color_val, color_val)
        cv2.arrowedLine(vis, (x, y), (ex, ey), color, 1, tipLength=0.3)

    return vis


def draw_density_overlay(frame_bgr: np.ndarray, density_map: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
    """Overlay density heatmap on frame."""
    H, W = frame_bgr.shape[:2]
    dmap = cv2.resize(density_map, (W, H))
    dmap_uint8 = (dmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(dmap_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap, alpha, 0)
    return blended


def draw_entropy_grid(frame_bgr: np.ndarray, entropy_map: np.ndarray,
                      alpha: float = 0.4) -> np.ndarray:
    """Draw directional entropy grid overlay."""
    vis = frame_bgr.copy()
    H, W = frame_bgr.shape[:2]
    rows, cols = entropy_map.shape
    cell_h = H // rows
    cell_w = W // cols
    max_e = entropy_map.max() + 1e-8

    for r in range(rows):
        for c in range(cols):
            e = entropy_map[r, c] / max_e
            color_intensity = int(e * 255)
            overlay_color = (0, color_intensity, 255 - color_intensity)
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), overlay_color, -1)
            vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 1)

    return vis


def save_metric_plot(values: list, title: str, ylabel: str,
                     save_path: str, xlabel: str = "Frame"):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(values, color="red", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)