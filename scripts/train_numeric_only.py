from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 256
HIDDEN_DIM = 64
DROPOUT = 0.1
LEARNING_RATE = 0.001
EPOCHS = 10
LABEL_COL = "team_a_win"

TRAIN_SUBSET_ROWS = None

NUMERIC_FEATURE_COLS = [
    # player-strength features
    "top_wr_diff",
    "jg_wr_diff",
    "mid_wr_diff",
    "adc_wr_diff",
    "sup_wr_diff",

    # confidence features
    "top_low_games_flag",
    "jg_low_games_flag",
    "mid_low_games_flag",
    "adc_low_games_flag",
    "sup_low_games_flag",
    "low_games_count",
    "avg_games_diff",
    # subclass counts
    "team_a_num_tanks",
    "team_a_num_bruisers",
    "team_a_num_mages",
    "team_a_num_marksmen",
    "team_a_num_assassins",
    "team_a_num_enchanters",
    "team_a_num_engages",

    "team_b_num_tanks",
    "team_b_num_bruisers",
    "team_b_num_mages",
    "team_b_num_marksmen",
    "team_b_num_assassins",
    "team_b_num_enchanters",
    "team_b_num_engages",

    # frontline / weird-role flags
    "team_a_frontline_count",
    "team_b_frontline_count",
    "team_a_top_is_enchanter",
    "team_b_top_is_enchanter",
    "team_a_jg_is_enchanter",
    "team_b_jg_is_enchanter",
]


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_latest_cleaned_csv(data_dir: Path) -> Path:
    candidates = list(data_dir.glob("*cleaned.csv"))
    if not candidates:
        raise FileNotFoundError(f"No cleaned CSV found in {data_dir.resolve()}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_cleaned_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Loaded empty CSV: {csv_path}")
    return df


def split_dataframe(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("Train/val/test fractions must sum to 1.")

    n = len(df)
    if n < 3:
        raise ValueError("Need at least 3 rows to make train/val/test split.")

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()

    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def take_subset(df: pd.DataFrame, n_rows: int | None) -> pd.DataFrame:
    if n_rows is None or n_rows >= len(df):
        return df.copy()
    return df.iloc[:n_rows].reset_index(drop=True)


def standardize_numeric_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0)

    train_df[feature_cols] = (train_df[feature_cols] - mean) / std
    val_df[feature_cols] = (val_df[feature_cols] - mean) / std
    test_df[feature_cols] = (test_df[feature_cols] - mean) / std

    return train_df, val_df, test_df, mean, std


# -----------------------------
# Dataset / Model
# -----------------------------
class NumericOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        numeric_features = torch.tensor(
            [float(row[col]) for col in NUMERIC_FEATURE_COLS],
            dtype=torch.float32,
        )
        label = torch.tensor(float(row[LABEL_COL]), dtype=torch.float32)

        return {
            "numeric_features": numeric_features,
            "label": label,
        }


class NumericOnlyMLP(nn.Module):
    def __init__(self, num_numeric_features: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(num_numeric_features),
            nn.Linear(num_numeric_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, numeric_features: torch.Tensor) -> torch.Tensor:
        return self.net(numeric_features).squeeze(-1)


def make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = NumericOnlyDataset(df)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        numeric_features = batch["numeric_features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(numeric_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        numeric_features = batch["numeric_features"].to(device)
        labels = batch["label"].to(device)

        logits = model(numeric_features)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_examples += batch_size

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


# -----------------------------
# Calibration
# -----------------------------
@torch.no_grad()
def collect_probs_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_probs = []
    all_labels = []

    for batch in loader:
        numeric_features = batch["numeric_features"].to(device)
        labels = batch["label"].to(device)

        logits = model(numeric_features)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    return probs_np, labels_np


def make_calibration_table(
    probs: np.ndarray,
    labels: np.ndarray,
    bin_width: float = 0.05,
) -> pd.DataFrame:
    edges = np.arange(0.0, 1.0 + bin_width, bin_width)
    edges[-1] = 1.0

    rows = []

    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]

        if i == len(edges) - 2:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)

        count = int(mask.sum())

        if count == 0:
            rows.append({
                "bucket": f"{int(left * 100):02d}-{int(right * 100):02d}%",
                "count": 0,
                "avg_pred_prob": np.nan,
                "actual_win_rate": np.nan,
            })
        else:
            rows.append({
                "bucket": f"{int(left * 100):02d}-{int(right * 100):02d}%",
                "count": count,
                "avg_pred_prob": float(probs[mask].mean()),
                "actual_win_rate": float(labels[mask].mean()),
            })

    return pd.DataFrame(rows)


def plot_calibration_table(calibration_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = calibration_df.dropna().copy()

    if plot_df.empty:
        print("No non-empty calibration buckets to plot.")
        return

    x = plot_df["avg_pred_prob"].to_numpy()
    y = plot_df["actual_win_rate"].to_numpy()

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(x, y, marker="o", label="Numeric-only model")
    plt.xlabel("Average predicted probability")
    plt.ylabel("Actual win rate")
    plt.title("Calibration plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_calibration_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_csv_path: Path,
    output_plot_path: Path,
    bin_width: float = 0.05,
) -> None:
    probs, labels = collect_probs_and_labels(model, loader, device)
    calibration_df = make_calibration_table(probs, labels, bin_width=bin_width)

    print("\nCalibration table:")
    print(calibration_df.to_string(index=False))

    calibration_df.to_csv(output_csv_path, index=False)
    plot_calibration_table(calibration_df, output_plot_path)

    print(f"\nSaved calibration table to: {output_csv_path.resolve()}")
    print(f"Saved calibration plot to:  {output_plot_path.resolve()}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    csv_path = find_latest_cleaned_csv(DATA_DIR)
    print(f"Loading dataset from: {csv_path.resolve()}")

    df = load_cleaned_csv(csv_path)

    train_df, val_df, test_df = split_dataframe(df, 0.70, 0.15, 0.15, seed=SEED)
    train_df, val_df, test_df, numeric_mean, numeric_std = standardize_numeric_features(
        train_df, val_df, test_df, NUMERIC_FEATURE_COLS
    )

    if TRAIN_SUBSET_ROWS is not None:
        train_df = take_subset(train_df, TRAIN_SUBSET_ROWS)

    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"Numeric features: {len(NUMERIC_FEATURE_COLS)}")

    train_loader = make_loader(train_df, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_df, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(test_df, BATCH_SIZE, shuffle=False)

    model = NumericOnlyMLP(
        num_numeric_features=len(NUMERIC_FEATURE_COLS),
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_model_path = OUTPUT_DIR / "best_numeric_only.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")

    calibration_csv_path = OUTPUT_DIR / "test_calibration_5pct_numeric_only.csv"
    calibration_plot_path = OUTPUT_DIR / "test_calibration_5pct_numeric_only.png"

    print_calibration_report(
        model=model,
        loader=test_loader,
        device=device,
        output_csv_path=calibration_csv_path,
        output_plot_path=calibration_plot_path,
        bin_width=0.05,
    )


if __name__ == "__main__":
    main()