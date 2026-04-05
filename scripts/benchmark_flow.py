import cv2
import yaml
import time
import numpy as np
from pathlib import Path
from src.flow import FlowExtractor
from src.utils import draw_flow_vectors, draw_density_overlay, draw_entropy_grid


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def benchmark(cfg, method: str, n_pairs: int = 30):
    cfg["flow"]["method"] = method
    extractor = FlowExtractor(cfg)

    W = cfg["data"]["resize"]["width"]
    H = cfg["data"]["resize"]["height"]

    times = []
    for _ in range(n_pairs):
        f1 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        f2 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        extractor.compute(f1, f2)
        times.append(time.perf_counter() - t0)

    avg = np.mean(times[5:])  # skip warmup
    fps = 1.0 / avg
    print(f"[{method:12s}] avg={avg*1000:.1f}ms  fps={fps:.1f}")
    return fps


def visualize_sample(cfg):
    """Run flow on two real frames from the dataset and save visualization."""
    frames_dir = Path(cfg["data"]["frames_dir"])
    clips = list(frames_dir.glob("*/*"))
    if not clips:
        print("No extracted frames found — run extract_frames.py first")
        return

    # Find a clip with at least 2 frames
    frame_files = None
    for clip in clips:
        flist = sorted(clip.glob("*.jpg"))
        if len(flist) >= 2:
            frame_files = flist
            break

    if frame_files is None:
        print("No valid clip found for visualization")
        return

    f1 = cv2.imread(str(frame_files[0]))
    f2 = cv2.imread(str(frame_files[1]))

    cfg["flow"]["method"] = "farneback"
    extractor = FlowExtractor(cfg)
    result = extractor.compute(f1, f2)

    vis_flow = draw_flow_vectors(f1, result["flow"], step=20, scale=2.0)
    vis_entropy = draw_entropy_grid(f1, result["directional_entropy"])

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "flow_vectors.jpg"), vis_flow)
    cv2.imwrite(str(out_dir / "entropy_grid.jpg"), vis_entropy)

    print(f"Flow vectors saved   : results/flow_vectors.jpg")
    print(f"Entropy grid saved   : results/entropy_grid.jpg")
    print(f"Magnitude map shape  : {result['magnitude_map'].shape}")
    print(f"Magnitude map range  : [{result['magnitude_map'].min():.4f}, {result['magnitude_map'].max():.4f}]")
    print(f"Entropy map shape    : {result['directional_entropy'].shape}")
    print(f"Entropy map range    : [{result['directional_entropy'].min():.4f}, {result['directional_entropy'].max():.4f}]")


if __name__ == "__main__":
    cfg = load_config()

    print("=== Flow Benchmark ===")
    fb_fps = benchmark(cfg, "farneback")
    raft_fps = benchmark(cfg, "raft")

    print(f"\nSelected method for pipeline: ", end="")
    if fb_fps >= 15:
        print("farneback (meets FPS target)")
    else:
        print("farneback (use frame skipping in inference)")

    print("\n=== Sample Visualization ===")
    visualize_sample(cfg)