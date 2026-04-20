from __future__ import annotations

from pathlib import Path
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.embedding_ids import load_cleaned_csv, build_champion_ids, DraftDataset, TEAM_A_COLS, TEAM_B_COLS, LABEL_COL, SCALING_TO_ID, SUBCLASS_TO_ID, TEAM_A_SCALING_COLS, TEAM_A_SUBCLASS_COLS, TEAM_B_SCALING_COLS, TEAM_B_SUBCLASS_COLS, NUMERIC_FEATURE_COLS
from src.model import DraftTransformer


# dir
DATA_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# hyperparams
BATCH_SIZE = 256
EMBED_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 3
FF_DIM = 64
DROPOUT = 0.1
MLP_HIDDEN_DIM = 64
LEARNING_RATE = 0.0003
EPOCHS = 10
SEED = 42

# seed
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

def find_latest_cleaned_csv(data_dir: Path) -> Path:
    """
    Find latest modified csv from /cleaned
    """
    candidates = list(data_dir.glob("*_cleaned.csv"))
    if not candidates:
        raise FileNotFoundError(f"No cleaned CSV found in {data_dir.resolve()}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def take_subset(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """ 
    Take the first n rows of a dataframe
    """
    if n_rows > len(df):
        raise ValueError(f"Requested {n_rows} rows, but only have {len(df)}")
    return df.iloc[:n_rows].reset_index(drop=True)

def split_dataframe(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Shuffle and split a dataframe into train/val/test, returns new train/val/test dataframes
    """

    # slight input error checking

    # sum to 1
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("Train/val/test fractions must sum to 1.")

    # atleaset 3 rows
    n = len(df)
    if n < 3:
        raise ValueError("Need at least 3 rows to make train/val/test split.")

    val_size = max(1, round(val_frac * n))
    test_size = max(1, round(test_frac * n))
    train_size = n - val_size - test_size

    #  val and test are at least 1, and thus checking on train is sufficient to ensure we can proceed
    if train_size < 1:
        raise ValueError(
            f"Split too small. Got train={train_size}, val={val_size}, test={test_size}."
        )

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()

    train_idx = perm[:train_size]
    val_idx = perm[train_size : train_size + val_size]
    test_idx = perm[train_size + val_size :]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df

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
    std = train_df[feature_cols].std()

    # avoid divide-by-zero 
    std = std.replace(0, 1.0)

    train_df[feature_cols] = (train_df[feature_cols] - mean) / std
    val_df[feature_cols] = (val_df[feature_cols] - mean) / std
    test_df[feature_cols] = (test_df[feature_cols] - mean) / std

    return train_df, val_df, test_df, mean, std

def make_loader(df: pd.DataFrame, champ_to_id: dict[str, int], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = DraftDataset(df, champ_to_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total

# calibration

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        if not np.any(mask):
            continue

        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_weight = mask.sum() / n

        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        if not np.any(mask):
            continue

        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        mce = max(mce, abs(bin_acc - bin_conf))

    return float(mce)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        champ_ids = batch["champ_ids"].to(device)
        team_ids = batch["team_ids"].to(device)
        role_ids = batch["role_ids"].to(device)
        subclass_ids = batch["subclass_ids"].to(device)
        scaling_ids = batch["scaling_ids"].to(device)
        numeric_features = batch["numeric_features"].to(device)
        labels = batch["label"].to(device)
        

        optimizer.zero_grad()
        logits = model(numeric_features, champ_ids, team_ids, role_ids, subclass_ids, scaling_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch in loader:
        champ_ids = batch["champ_ids"].to(device)
        team_ids = batch["team_ids"].to(device)
        role_ids = batch["role_ids"].to(device)
        numeric_features = batch["numeric_features"].to(device)
        subclass_ids = batch["subclass_ids"].to(device)
        scaling_ids = batch["scaling_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(numeric_features, champ_ids, team_ids, role_ids, subclass_ids, scaling_ids)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def print_example_predictions(
    model: nn.Module,
    df: pd.DataFrame,
    champ_to_id: dict[str, int],
    device: torch.device,
    num_examples: int = 5,
) -> None:
    print("\nExample predictions:")

    for i in range(min(num_examples, len(df))):
        row = df.iloc[i]

        team_a = [str(row[col]).strip() for col in TEAM_A_COLS]
        team_b = [str(row[col]).strip() for col in TEAM_B_COLS]

        champ_ids = torch.tensor(
            [[champ_to_id[c] for c in team_a + team_b]],
            dtype=torch.long,
            device=device,
        )

        team_ids = torch.tensor(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]],
            dtype=torch.long,
            device=device,
        )

        role_ids = torch.tensor(
            [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]],
            dtype=torch.long,
            device=device,
        )

        subclass_cols = TEAM_A_SUBCLASS_COLS + TEAM_B_SUBCLASS_COLS
        scaling_cols = TEAM_A_SCALING_COLS + TEAM_B_SCALING_COLS

        subclass_ids = torch.tensor(
            [[SUBCLASS_TO_ID[str(row[col]).strip()] for col in subclass_cols]],
            dtype=torch.long,
            device=device,
        )

        scaling_ids = torch.tensor(
            [[SCALING_TO_ID[str(row[col]).strip()] for col in scaling_cols]],
            dtype=torch.long,
            device=device,
        )

        numeric_features = torch.tensor(
            [[float(row[col]) for col in NUMERIC_FEATURE_COLS]],
            dtype=torch.float32,
            device=device,
        )

        logit = model(
            numeric_features,
            champ_ids,
            team_ids,
            role_ids,
            subclass_ids,
            scaling_ids,
        )

        prob = torch.sigmoid(logit).item()
        pred = int(prob >= 0.5)
        actual = int(row[LABEL_COL])

        print(f"Row {i:02d}")
        print(f"  Team A: {team_a}")
        print(f"  Team B: {team_b}")
        print(f"  Actual team_a_win: {actual}")
        print(f"  Predicted prob(team_a_win=1): {prob:.4f}")
        print(f"  Predicted class: {pred}")

def plot_losses(train_losses: list[float], val_losses: list[float], out_path: Path) -> None:
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Calibration functions 
def collect_probs_and_labels(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            champ_ids = batch["champ_ids"].to(device)
            team_ids = batch["team_ids"].to(device)
            role_ids = batch["role_ids"].to(device)
            subclass_ids = batch["subclass_ids"].to(device)
            scaling_ids = batch["scaling_ids"].to(device)
            numeric_features = batch["numeric_features"].to(device)
            labels = batch["label"].float().to(device)

            logits = model(numeric_features, champ_ids, team_ids, role_ids, subclass_ids, scaling_ids)
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
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length.")

    edges = np.arange(0.0, 1.0 + bin_width, bin_width)
    edges[-1] = 1.0  # avoid float weirdness

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
                "bucket_left": left,
                "bucket_right": right,
                "count": 0,
                "avg_pred_prob": np.nan,
                "actual_win_rate": np.nan,
            })
        else:
            bucket_probs = probs[mask]
            bucket_labels = labels[mask]

            rows.append({
                "bucket_left": left,
                "bucket_right": right,
                "count": count,
                "avg_pred_prob": float(bucket_probs.mean()),
                "actual_win_rate": float(bucket_labels.mean()),
            })

    table = pd.DataFrame(rows)
    table["bucket"] = table.apply(
        lambda r: f"{int(r['bucket_left'] * 100):02d}-{int(r['bucket_right'] * 100):02d}%",
        axis=1,
    )

    return table[
        ["bucket", "count", "avg_pred_prob", "actual_win_rate"]
    ]


def plot_calibration_table(
    calibration_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = calibration_df.dropna().copy()

    if plot_df.empty:
        print("No non-empty calibration buckets to plot.")
        return

    x = plot_df["avg_pred_prob"].to_numpy()
    y = plot_df["actual_win_rate"].to_numpy()

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(x, y, marker="o", label="Model")
    plt.xlabel("Average predicted probability")
    plt.ylabel("Actual win rate")
    plt.title("Calibration plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_calibration_report(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
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
    champ_to_id = build_champion_ids(df)

    train_df, val_df, test_df = split_dataframe(df, 0.70, 0.15, 0.15, seed=SEED)
    train_df, val_df, test_df, numeric_mean, numeric_std = standardize_numeric_features(
    train_df, val_df, test_df, NUMERIC_FEATURE_COLS
    )
    
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"Unique champions: {len(champ_to_id)}")

    train_loader = make_loader(train_df, champ_to_id, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_df, champ_to_id, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(test_df, champ_to_id, BATCH_SIZE, shuffle=False)

    model = DraftTransformer(
        num_champions=len(champ_to_id),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        num_numeric_features=len(NUMERIC_FEATURE_COLS),
    ) .to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay = 0.001)

    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val_loss = float("inf")
    best_model_path = OUTPUT_DIR / "best_draft_transformer.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    # load best checkpoint before testing
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\nLoaded best model from: {best_model_path.resolve()} with val loss: {best_val_loss:.4f}")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")

    loss_plot_path = OUTPUT_DIR / "loss_curve.png"
    plot_losses(train_losses, val_losses, loss_plot_path)
    print(f"Saved loss curve to: {loss_plot_path.resolve()}")
    calibration_csv_path = OUTPUT_DIR / "test_calibration_5pct.csv"
    calibration_plot_path = OUTPUT_DIR / "test_calibration_5pct.png"

    print_calibration_report(
        model=model,
        loader=test_loader,
        device=device,
        output_csv_path=calibration_csv_path,
        output_plot_path=calibration_plot_path,
        bin_width=0.05,
    )

    # scalar calibration metrics
    test_probs, test_labels = collect_probs_and_labels(model, test_loader, device)

    ece_10 = expected_calibration_error(test_labels, test_probs, n_bins=10)
    ece_20 = expected_calibration_error(test_labels, test_probs, n_bins=20)
    mce_10 = maximum_calibration_error(test_labels, test_probs, n_bins=10)
    brier = brier_score(test_labels, test_probs)

    print(f"\nTest ECE (10 bins): {ece_10:.4f}")
    print(f"Test ECE (20 bins): {ece_20:.4f}")
    print(f"Test MCE (10 bins): {mce_10:.4f}")
    print(f"Test Brier Score: {brier:.4f}")

    metrics_path = OUTPUT_DIR / "test_calibration_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Test ECE (10 bins): {ece_10:.6f}\n")
        f.write(f"Test ECE (20 bins): {ece_20:.6f}\n")
        f.write(f"Test MCE (10 bins): {mce_10:.6f}\n")
        f.write(f"Test Brier Score: {brier:.6f}\n")

    print(f"Saved calibration metrics to: {metrics_path.resolve()}")

    print_example_predictions(model, test_df, champ_to_id, device, num_examples=5)


if __name__ == "__main__":
    main()