# Project V3.2: Crowd Anomaly and Panic Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS-orange)
![AUC](https://img.shields.io/badge/AUC--ROC-0.9648-green)

A dual-stream crowd anomaly detection system combining appearance density maps
and optical flow motion signals, with a ConvLSTM temporal classifier and an
automated alert + clip extraction pipeline.

---

## Architecture
```
Input Video
    │
    ├── Density Stream (CSRNet / Edge Proxy) ──┐
    │                                           ├─→ [T × (density, flow)] → ConvLSTM → Anomaly Score
    └── Flow Stream (Farneback Optical Flow) ──┘
                                                         │
                                              ┌──────────▼──────────┐
                                              │   Alert System       │
                                              │  threshold=0.6       │
                                              │  auto clip extract   │
                                              └─────────────────────┘
```

## Results

| Metric    | Value  |
|-----------|--------|
| AUC-ROC   | 0.9648 |
| Precision | 0.8952 |
| Recall    | 0.8162 |
| F1        | 0.8538 |
| Accuracy  | 0.9422 |
| Pipeline FPS | ~30 (M1 MPS) |

## Dataset

**ShanghaiTech Campus Dataset** — 13 scenes, 130 anomaly events, 270K+ frames.
- Training: 330 normal video clips
- Testing: 107 clips with frame-level anomaly annotations

## Tech Stack

- **Framework**: PyTorch (MPS inference on M1)
- **Temporal Model**: ConvLSTM classifier
- **Optical Flow**: Farneback (OpenCV) + RAFT (torchvision)
- **Density Stream**: CSRNet (V3.1 weights) / Edge proxy
- **Training**: Google Colab T4
- **Environment**: conda

## Project Structure
```
├── configs/config.yaml         # All parameters — no hardcoded values
├── src/
│   ├── pipeline.py             # Unified inference pipeline
│   ├── density.py              # CSRNet density stream
│   ├── flow.py                 # Optical flow extraction
│   ├── model.py                # ConvLSTM / 3D CNN classifier
│   ├── alert.py                # Alert system + clip extraction
│   └── utils.py                # Visualization utilities
├── scripts/
│   ├── extract_frames.py       # Dataset preprocessing
│   ├── prepare_windows.py      # Sliding window feature generation
│   ├── benchmark_flow.py       # Flow method benchmarking
│   ├── evaluate.py             # Full evaluation pipeline
│   ├── run_inference.py        # Full inference pipeline
│   └── burn_labels.py          # Alert clip label burning
├── notebooks/
│   └── train_classifier.ipynb  # Colab training notebook
└── results/
    ├── metrics.csv
    ├── roc_curve.png
    ├── confusion_matrix.png
    └── score_distribution.png
```

## Setup
```bash
conda create -n v32_crowd python=3.10 -y
conda activate v32_crowd
pip install torch torchvision torchaudio
pip install opencv-python opencv-contrib-python
pip install matplotlib numpy pandas scipy scikit-learn tqdm pyyaml h5py gdown
```

## Usage

### Run inference on a video
```bash
python scripts/run_inference.py --video path/to/video.mp4
```

### Evaluate classifier
```bash
python scripts/evaluate.py
```

### Benchmark optical flow
```bash
python -m scripts.benchmark_flow
```

## Training

Training runs on Google Colab T4. See `notebooks/train_classifier.ipynb`.
Dataset: ShanghaiTech Campus — windows generated via `scripts/prepare_windows.py`.

## Demo

- **Full annotated video**: dual overlay (density heatmap + flow vectors) with
  anomaly score timeline and HUD
- **Alert clips**: auto-extracted ±3s around each anomaly trigger with event label

## Citation
```
@INPROCEEDINGS{liu2018ano_pred,
  author={W. Liu and W. Luo and D. Lian and S. Gao},
  title={Future Frame Prediction for Anomaly Detection -- A New Baseline},
  booktitle={CVPR},
  year={2018}
}
```