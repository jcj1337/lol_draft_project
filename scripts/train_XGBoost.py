from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from src.embedding_ids import load_cleaned_csv, build_champion_ids, DraftDataset, LABEL_COL, SCALING_TO_ID, SUBCLASS_TO_ID, TEAM_A_SCALING_COLS, TEAM_A_SUBCLASS_COLS, TEAM_B_SCALING_COLS, TEAM_B_SUBCLASS_COLS, NUMERIC_FEATURE_COLS

# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("outputs/xgboost")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# set to None for full train split
TRAIN_SUBSET_ROWS = None

# set to False for one run
RUN_GRID_SEARCH = True

CATEGORICAL_FEATURE_COLS = [
    "patch",
    "team_a_top",
    "team_a_jg",
    "team_a_mid",
    "team_a_adc",
    "team_a_sup",
    "team_b_top",
    "team_b_jg",
    "team_b_mid",
    "team_b_adc",
    "team_b_sup",
]
# small grid to start
SEARCH_SPACE = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def take_subset(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if n_rows is None or n_rows >= len(df):
        return df.copy()
    return df.iloc[:n_rows].reset_index(drop=True)


def validate_columns(df: pd.DataFrame) -> None:
    needed = CATEGORICAL_FEATURE_COLS + NUMERIC_FEATURE_COLS + [LABEL_COL]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df[CATEGORICAL_FEATURE_COLS + NUMERIC_FEATURE_COLS].copy()
    y = df[LABEL_COL].astype(int).copy()
    return x, y


def build_pipeline(config: dict) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURE_COLS,
            ),
            (
                "num",
                "passthrough",
                NUMERIC_FEATURE_COLS,
            ),
        ]
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        tree_method="hist",
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def evaluate_split(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[float, float]:
    probs = pipeline.predict_proba(x)[:, 1]
    preds = (probs >= 0.5).astype(int)

    loss = log_loss(y, probs)
    acc = accuracy_score(y, preds)
    return loss, acc


def compute_calibration_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float]:
    """
    Returns:
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
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

        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        gap = abs(bin_acc - bin_conf)

        ece += (mask.sum() / n) * gap
        mce = max(mce, gap)

    return ece, mce


def get_calibration_metrics(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    probs = pipeline.predict_proba(x)[:, 1]

    ece_10, mce_10 = compute_calibration_metrics(y, probs, n_bins=10)
    ece_20, _ = compute_calibration_metrics(y, probs, n_bins=20)
    brier = brier_score_loss(y, probs)

    return {
        "Test ECE (10 bins)": ece_10,
        "Test ECE (20 bins)": ece_20,
        "Test MCE (10 bins)": mce_10,
        "Test Brier Score": brier,
    }


def save_calibration_metrics(metrics: dict[str, float], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")


def print_example_predictions(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    num_examples: int = 5,
) -> None:
    sample_df = test_df.head(num_examples).copy()
    x_sample, y_sample = make_xy(sample_df)
    probs = pipeline.predict_proba(x_sample)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\nExample predictions:")
    for i in range(len(sample_df)):
        print(
            f"Row {i:02d} | "
            f"True: {int(y_sample.iloc[i])} | "
            f"Pred: {int(preds[i])} | "
            f"Prob(team_a_win=1): {probs[i]:.4f}"
        )


def run_one_config(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    x_train, y_train = make_xy(train_df)
    x_val, y_val = make_xy(val_df)

    pipeline = build_pipeline(config)
    pipeline.fit(x_train, y_train)

    train_loss, train_acc = evaluate_split(pipeline, x_train, y_train)
    val_loss, val_acc = evaluate_split(pipeline, x_val, y_val)

    result = {
        **config,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    return pipeline, result


def main() -> None:
    set_seed(SEED)

    csv_path = find_latest_cleaned_csv(DATA_DIR)
    print(f"Loading dataset from: {csv_path.resolve()}")

    df = load_cleaned_csv(csv_path)
    validate_columns(df)

    train_df, val_df, test_df = split_dataframe(df, 0.70, 0.15, 0.15, seed=SEED)

    if TRAIN_SUBSET_ROWS is not None:
        train_df = take_subset(train_df, TRAIN_SUBSET_ROWS)

    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"Categorical feature cols: {len(CATEGORICAL_FEATURE_COLS)}")
    print(f"Numeric feature cols:     {len(NUMERIC_FEATURE_COLS)}")

    if RUN_GRID_SEARCH:
        param_names = list(SEARCH_SPACE.keys())
        param_values = [SEARCH_SPACE[name] for name in param_names]
        configs = [dict(zip(param_names, vals)) for vals in product(*param_values)]

        print(f"\nRunning {len(configs)} XGBoost configurations...\n")

        results = []
        best_pipeline = None
        best_result = None

        for i, config in enumerate(configs, start=1):
            print(f"=== Config {i}/{len(configs)} ===")
            pipeline, result = run_one_config(config, train_df, val_df)

            print(
                f"Train Loss: {result['train_loss']:.4f} | "
                f"Train Acc: {result['train_acc']:.4f} | "
                f"Val Loss: {result['val_loss']:.4f} | "
                f"Val Acc: {result['val_acc']:.4f}"
            )

            results.append(result)

            if best_result is None or result["val_loss"] < best_result["val_loss"]:
                best_result = result
                best_pipeline = pipeline

            pd.DataFrame(results).sort_values(
                by=["val_loss", "val_acc"],
                ascending=[True, False],
            ).to_csv(OUTPUT_DIR / "xgboost_results.csv", index=False)

        assert best_pipeline is not None
        assert best_result is not None

        print("\n=== Best Validation Result ===")
        print(json.dumps(best_result, indent=2))

        x_test, y_test = make_xy(test_df)
        test_loss, test_acc = evaluate_split(best_pipeline, x_test, y_test)

        print(f"\nFinal Test Loss: {test_loss:.4f}")
        print(f"Final Test Accuracy: {test_acc:.4f}")

        calibration_metrics = get_calibration_metrics(best_pipeline, x_test, y_test)
        for key, value in calibration_metrics.items():
            print(f"{key}: {value:.6f}")

        save_calibration_metrics(
            calibration_metrics,
            OUTPUT_DIR / "test_calibration_metrics.txt",
        )

        with open(OUTPUT_DIR / "best_xgboost_config.json", "w", encoding="utf-8") as f:
            json.dump(best_result, f, indent=2)

        print_example_predictions(best_pipeline, test_df, num_examples=5)

        print(f"\nSaved results to: {(OUTPUT_DIR / 'xgboost_results.csv').resolve()}")
        print(f"Saved best config to: {(OUTPUT_DIR / 'best_xgboost_config.json').resolve()}")
        print(
            f"Saved calibration metrics to: "
            f"{(OUTPUT_DIR / 'test_calibration_metrics.txt').resolve()}"
        )

    else:
        config = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        pipeline, result = run_one_config(config, train_df, val_df)

        print("\n=== Single Run ===")
        print(
            f"Train Loss: {result['train_loss']:.4f} | "
            f"Train Acc: {result['train_acc']:.4f} | "
            f"Val Loss: {result['val_loss']:.4f} | "
            f"Val Acc: {result['val_acc']:.4f}"
        )

        x_test, y_test = make_xy(test_df)
        test_loss, test_acc = evaluate_split(pipeline, x_test, y_test)

        print(f"\nFinal Test Loss: {test_loss:.4f}")
        print(f"Final Test Accuracy: {test_acc:.4f}")

        calibration_metrics = get_calibration_metrics(pipeline, x_test, y_test)
        for key, value in calibration_metrics.items():
            print(f"{key}: {value:.6f}")

        save_calibration_metrics(
            calibration_metrics,
            OUTPUT_DIR / "test_calibration_metrics.txt",
        )

        print_example_predictions(pipeline, test_df, num_examples=5)

        print(
            f"\nSaved calibration metrics to: "
            f"{(OUTPUT_DIR / 'test_calibration_metrics.txt').resolve()}"
        )


if __name__ == "__main__":
    main()