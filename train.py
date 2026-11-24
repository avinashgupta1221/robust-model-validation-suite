"""Robust neural-network validation and regularization experiments."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
RESULTS.mkdir(exist_ok=True)
REPORT.mkdir(exist_ok=True)
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class Network(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(30, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)


def get_data():
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target.astype(np.float32)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    def tensor(x, y):
        return TensorDataset(torch.tensor(x, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32).reshape(-1, 1))
    return tensor(X_train, y_train), tensor(X_val, y_val), tensor(X_test, y_test)


def train_one(name, dropout=0.0, weight_decay=0.0, early_stop=False, epochs=150):
    seed_everything()
    train_ds, val_ds, test_ds = get_data()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    model = Network(dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf"); best_state = None; patience, wait = 15, 0

    for epoch in range(epochs):
        model.train(); total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); optimizer.step(); total += loss.item() * len(xb)
        train_loss = total / len(train_ds)
        model.eval(); total_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                total_val += loss_fn(model(xb), yb).item() * len(xb)
        val_loss = total_val / len(val_ds)
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        if val_loss < best_val - 1e-4:
            best_val = val_loss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if early_stop and wait >= patience:
            print(f"{name}: early stopping at epoch {epoch+1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_prob = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            y_true.extend(yb.numpy().ravel()); y_prob.extend(torch.sigmoid(model(xb)).numpy().ravel())
    y_pred = (np.array(y_prob) >= .5).astype(int)
    metrics = {
        "Model": name,
        "Test Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Best Validation Loss": best_val,
        "Epochs Trained": len(history["train_loss"])
    }
    return metrics, history


def main():
    experiments = [
        ("Baseline", 0.0, 0.0, False),
        ("Dropout (0.30)", 0.30, 0.0, False),
        ("L2 Weight Decay (0.001)", 0.0, 0.001, False),
        ("Early Stopping", 0.0, 0.0, True),
    ]
    rows, histories = [], {}
    for name, dropout, wd, es in experiments:
        print(f"Training {name}...")
        metrics, hist = train_one(name, dropout, wd, es)
        rows.append(metrics); histories[name] = hist
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "model_comparison.csv", index=False)

    # Learning-curve plots: each model's train and validation loss.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for ax, (name, hist) in zip(axes.ravel(), histories.items()):
        ax.plot(hist["train_loss"], label="Training loss")
        ax.plot(hist["val_loss"], label="Validation loss")
        ax.set_title(name); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(alpha=.25); ax.legend()
    fig.suptitle("Learning Curves: Regularization Experiments", fontsize=15)
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curves.png", dpi=160)
    plt.close(fig)

    # Test metric comparison.
    metrics_cols = ["Test Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ax = df.set_index("Model")[metrics_cols].plot(kind="bar", figsize=(12, 6))
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Test-set Performance Comparison")
    ax.grid(axis="y", alpha=.25); plt.xticks(rotation=15, ha="right"); plt.tight_layout()
    plt.savefig(RESULTS / "metric_comparison.png", dpi=160); plt.close()

    best = df.loc[df["F1 Score"].idxmax()]
    report = f"""# Experiment Report: Model Validation and Regularization\n\n## Dataset and validation setup\nThe Wisconsin Breast Cancer dataset from scikit-learn was used (569 samples, 30 numeric features, binary target). Data was split into 70% training, 15% validation, and 15% test sets using stratification and a fixed random seed (42). Standardization was fitted on the training data only, then applied to validation and test sets.\n\n## Models\n- **Baseline:** two hidden layers (64 and 32 units), ReLU, Adam, no regularization.\n- **Dropout:** dropout probability 0.30 between hidden layers.\n- **L2:** Adam weight decay of 0.001.\n- **Early stopping:** baseline architecture; training stops when validation loss does not improve for 15 epochs. The best validation-loss weights are restored for evaluation.\n\n## Results\nSee `results/model_comparison.csv` for full test metrics, `learning_curves.png` for training/validation loss, and `metric_comparison.png` for test scores. The model with the highest test F1 in this run was **{best['Model']}** (F1 = {best['F1 Score']:.4f}, accuracy = {best['Test Accuracy']:.4f}, ROC-AUC = {best['ROC-AUC']:.4f}).\n\n## Interpretation\nRegularization can reduce the gap between training and validation performance by discouraging overly complex fits. Dropout randomly disables hidden units during training; L2 weight decay penalizes large weights; early stopping limits training when validation performance stops improving. The best technique depends on the dataset, and the test set is reserved for final comparison rather than tuning. With this relatively small, clean dataset, differences may be modest and scores may be high across models.\n\n## Conclusion\nThe suite demonstrates a leakage-aware train/validation/test workflow, baseline learning curves, three regularization strategies, and consistent test-set metrics. For deployment, model choice should consider validation stability, test performance, and the cost of false positives/false negatives—not accuracy alone.\n"""
    (REPORT / "experiment_report.md").write_text(report, encoding="utf-8")
    print("\nResults saved in results/ and report/.")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
