from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import random
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.embedding_ids import (
    load_cleaned_csv,
    build_champion_ids,
    DraftDataset,
    NUMERIC_FEATURE_COLS,
)
from src.model import DraftTransformer


DATA_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("outputs/grid_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 10
DROPOUT = 0.1

# fixed hyperparameters (pointless to change)
TRAIN_SUBSET_ROWS = None
NUM_LAYERS = 2
FF_DIM = 64
MLP_HIDDEN_DIM = 64
DROPOUT = 0.1
NUM_HEADS = 2
EPOCHS = 10

# focused search
SEARCH_SPACE = {
    "batch_size": [128, 256],
    "learning_rate": [3e-4, 1e-3],
    "weight_decay": [0.0, 1e-4],
    "embed_dim": [32, 64],
    "num_layers": [2, 3],
}


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


def take_subset(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if n_rows >= len(df):
        return df.copy()
    return df.iloc[:n_rows].reset_index(drop=True)


def make_loader(
    df: pd.DataFrame,
    champ_to_id: dict[str, int],
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    dataset = DraftDataset(df=df, champ_to_id=champ_to_id)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        champ_ids = batch["champ_ids"].to(device)
        team_ids = batch["team_ids"].to(device)
        role_ids = batch["role_ids"].to(device)
        subclass_ids = batch["subclass_ids"].to(device)
        scaling_ids = batch["scaling_ids"].to(device)
        numeric_features = batch["numeric_features"].to(device)
        labels = batch["label"].float().to(device)

        optimizer.zero_grad()
        logits = model(
            numeric_features,
            champ_ids,
            team_ids,
            role_ids,
            subclass_ids,
            scaling_ids,
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_examples += bs

    return total_loss / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        champ_ids = batch["champ_ids"].to(device)
        team_ids = batch["team_ids"].to(device)
        role_ids = batch["role_ids"].to(device)
        subclass_ids = batch["subclass_ids"].to(device)
        scaling_ids = batch["scaling_ids"].to(device)
        numeric_features = batch["numeric_features"].to(device)
        labels = batch["label"].float().to(device)

        logits = model(
            numeric_features,
            champ_ids,
            team_ids,
            role_ids,
            subclass_ids,
            scaling_ids,
        )
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (preds == labels).sum().item()
        total_examples += bs

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def build_model(
    champ_to_id: dict[str, int],
    embed_dim: int,
    num_layers: int,
) -> DraftTransformer:
    return DraftTransformer(
        num_champions=len(champ_to_id),
        num_numeric_features=len(NUMERIC_FEATURE_COLS),
        embed_dim=embed_dim,
        num_heads=NUM_HEADS,
        num_layers=num_layers,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
    )


def run_one_config(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    champ_to_id: dict[str, int],
    device: torch.device,
) -> dict:
    train_loader = make_loader(train_df, champ_to_id, config["batch_size"], shuffle=True)
    val_loader = make_loader(val_df, champ_to_id, config["batch_size"], shuffle=False)

    model = build_model(
        champ_to_id=champ_to_id,
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc

        print(
            f"[{config}] "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return {
        **config,
        "dropout": DROPOUT,
        "ff_dim": FF_DIM,
        "mlp_hidden_dim": MLP_HIDDEN_DIM,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    }


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
    train_df, val_df, test_df, _, _ = standardize_numeric_features(
        train_df, val_df, test_df, NUMERIC_FEATURE_COLS
    )

    if TRAIN_SUBSET_ROWS is not None:
        train_df = take_subset(train_df, TRAIN_SUBSET_ROWS)

    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"Unique champions: {len(champ_to_id)}")

    
    param_names = list(SEARCH_SPACE.keys())
    param_values = [SEARCH_SPACE[name] for name in param_names]
    configs = [dict(zip(param_names, vals)) for vals in product(*param_values)]

    print(f"\nRunning {len(configs)} configurations...\n")

    results = []
    best_result = None
    search_start_time = time.time()
    for i, config in enumerate(configs, start=1):
        
        config_start_time = time.time()

        print(f"\n=== Config {i}/{len(configs)} ===")
        result = run_one_config(
            config=config,
            train_df=train_df,
            val_df=val_df,
            champ_to_id=champ_to_id,
            device=device,
        )

        config_elapsed = time.time() - config_start_time
        total_elapsed = time.time() - search_start_time
        avg_time_per_config = total_elapsed / i
        remaining_configs = len(configs) - i
        est_remaining = avg_time_per_config * remaining_configs
        est_total = avg_time_per_config * len(configs)

        print(
            f"Config time: {timedelta(seconds=int(config_elapsed))} | "
            f"Elapsed: {timedelta(seconds=int(total_elapsed))} | "
            f"Est. remaining: {timedelta(seconds=int(est_remaining))} | "
            f"Est. total: {timedelta(seconds=int(est_total))}"
        )

    results.append(result)

    if best_result is None or result["best_val_loss"] < best_result["best_val_loss"]:
        best_result = result

    pd.DataFrame(results).sort_values(
        by=["best_val_loss", "best_val_acc"],
        ascending=[True, False],
    ).to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False)

    assert best_result is not None

    with open(OUTPUT_DIR / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2)
    total_search_time = time.time() - search_start_time
    print(f"\nTotal grid search time: {timedelta(seconds=int(total_search_time))}")
    print("\n=== Best Result ===")
    print(json.dumps(best_result, indent=2))
    print(f"\nSaved results to: {(OUTPUT_DIR / 'grid_search_results.csv').resolve()}")
    print(f"Saved best config to: {(OUTPUT_DIR / 'best_config.json').resolve()}")


if __name__ == "__main__":
    main()