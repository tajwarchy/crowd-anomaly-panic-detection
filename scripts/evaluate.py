import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, RocCurveDisplay,
    precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from src.model import load_model


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class WindowDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["window_path"].tolist()
        self.labels = self.df["label"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        window = np.load(str(self.paths[int(idx)]))
        label = int(self.labels[int(idx)])
        return torch.from_numpy(window).float(), torch.tensor(label, dtype=torch.long)
    


@torch.no_grad()
def run_inference(model, loader):
    all_probs, all_preds, all_labels = [], [], []
    for x, y in loader:
        out = model(x)
        probs = torch.softmax(out, dim=1)[:, 1].numpy()
        preds = out.argmax(1).numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def get_test_df(df, seed=42):
    idx = np.arange(len(df))
    _, temp_idx = train_test_split(
        idx, test_size=0.3,
        stratify=df["label"].values, random_state=seed
    )
    _, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=df["label"].values[temp_idx], random_state=seed
    )
    return df.iloc[test_idx].reset_index(drop=True)


def evaluate(cfg):
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(exist_ok=True)

    df = pd.read_csv(cfg["data"]["manifest_path"])
    test_df = get_test_df(df)

    print(f"Test windows : {len(test_df)}")
    print(f"Label dist   : {test_df['label'].value_counts().to_dict()}")

    loader = DataLoader(
        WindowDataset(test_df),
        batch_size=32, shuffle=False, num_workers=0
    )

    model = load_model(cfg)
    labels, preds, probs = run_inference(model, loader)

    # --- Metrics ---
    auc = roc_auc_score(labels, probs)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds)

    print(f"\n{'='*40}")
    print(f"AUC-ROC    : {auc:.4f}")
    print(f"Precision  : {p:.4f}")
    print(f"Recall     : {r:.4f}")
    print(f"F1         : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=["normal", "anomaly"]))
    print(f"Confusion Matrix:\n{cm}")

    metrics = {"auc": auc, "precision": p, "recall": r, "f1": f1}
    pd.DataFrame([metrics]).to_csv(results_dir / "metrics.csv", index=False)

    # --- ROC Curve ---
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(labels, probs, ax=ax, name="ConvLSTM")
    ax.set_title("ROC Curve — Test Set")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(results_dir / "roc_curve.png"), dpi=120)
    plt.close()
    print(f"\nROC curve saved              : results/roc_curve.png")

    # --- Confusion Matrix Plot ---
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["normal", "anomaly"])
    ax.set_yticklabels(["normal", "anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(str(results_dir / "confusion_matrix.png"), dpi=120)
    plt.close()
    print(f"Confusion matrix saved       : results/confusion_matrix.png")

    # --- Score Distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[labels == 0], bins=40, alpha=0.6, label="normal",  color="steelblue")
    ax.hist(probs[labels == 1], bins=40, alpha=0.6, label="anomaly", color="crimson")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(results_dir / "score_distribution.png"), dpi=120)
    plt.close()
    print(f"Score distribution saved     : results/score_distribution.png")

    return metrics


if __name__ == "__main__":
    cfg = load_config()
    evaluate(cfg)