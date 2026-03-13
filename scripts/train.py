from __future__ import annotations

from pathlib import Path
import sys
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.embedding_ids import load_cleaned_csv, build_champion_ids, DraftDataset
from src.model import DraftTransformer


# dir
DATA_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# hyperparams
BATCH_SIZE =8
EMBED_DIM = 32
NUM_HEADS = 2
NUM_LAYERS = 2
FF_DIM = 64
DROPOUT = 0.2
MLP_HIDDEN_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 5
SEED = 42
# asdf asd f
TRAIN_SUBSET_ROWS = 240
# don't change these ever
TEAM_A_COLS = ["team_a_top", "team_a_jg", "team_a_mid", "team_a_adc", "team_a_sup"]
TEAM_B_COLS = ["team_b_top", "team_b_jg", "team_b_mid", "team_b_adc", "team_b_sup"]
LABEL_COL = "team_a_win"

# seed
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

def find_latest_cleaned_csv(data_dir: Path) -> Path:
    candidates = list(data_dir.glob("*_cleaned.csv"))
    if not candidates:
        raise FileNotFoundError(f"No cleaned CSV found in {data_dir.resolve()}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def take_subset(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if n_rows > len(df):
        raise ValueError(f"Requested {n_rows} rows, but only have {len(df)}")
    return df.iloc[:n_rows].reset_index(drop=True)

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

    # for too small
    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Split too small. Got train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def make_loader(df: pd.DataFrame, champ_to_id: dict[str, int], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = DraftDataset(df, champ_to_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total


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
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        role_ids = batch["role_ids"].to(device)
        logits = model(champ_ids, team_ids, role_ids)
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
        labels = batch["label"].to(device)

        role_ids = batch["role_ids"].to(device)
        logits = model(champ_ids, team_ids, role_ids)
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
    test_df: pd.DataFrame,
    champ_to_id: dict[str, int],
    device: torch.device,
    num_examples: int = 5,
) -> None:
    model.eval()

    print("\nExample predictions:")
    num_examples = min(num_examples, len(test_df))

    for i in range(num_examples):
        row = test_df.iloc[i]

        team_a = [str(row[col]).strip() for col in TEAM_A_COLS]
        team_b = [str(row[col]).strip() for col in TEAM_B_COLS]
        label = int(row[LABEL_COL])

        champ_ids = torch.tensor(
    [[champ_to_id[c] for c in team_a + team_b]],
    dtype=torch.long,
    device=device,
)
        team_ids = torch.tensor(
            [[0] * 5 + [1] * 5],
            dtype=torch.long,
            device=device,
        )
        role_ids = torch.tensor(
            [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]],
            dtype=torch.long,
            device=device,
        )

        logit = model(champ_ids, team_ids, role_ids)
        prob = torch.sigmoid(logit).item()
        pred = int(prob >= 0.5)

        print(f"\nExample {i + 1}")
        print(f"Team A: {team_a}")
        print(f"Team B: {team_b}")
        print(f"Actual team_a_win: {label}")
        print(f"Predicted prob(team_a_win=1): {prob:.4f}")
        print(f"Predicted class: {pred}")


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


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = find_latest_cleaned_csv(DATA_DIR)
    print(f"Loading dataset from: {csv_path.resolve()}")

    df = load_cleaned_csv(csv_path)
    champ_to_id = build_champion_ids(df)

    train_df, val_df, test_df = split_dataframe(df, 0.70, 0.15, 0.15, seed=SEED)

    # tester asdf 
    if TRAIN_SUBSET_ROWS is not None:
        train_df = take_subset(train_df, TRAIN_SUBSET_ROWS)

    # tester end 
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
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")

    loss_plot_path = OUTPUT_DIR / "loss_curve.png"
    plot_losses(train_losses, val_losses, loss_plot_path)
    print(f"Saved loss curve to: {loss_plot_path.resolve()}")

    print_example_predictions(model, test_df, champ_to_id, device, num_examples=5)


if __name__ == "__main__":
    main()