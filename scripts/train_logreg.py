from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("outputs/logreg")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
LABEL_COL = "team_a_win"

# set to None for full train split
TRAIN_SUBSET_ROWS = None

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

NUMERIC_FEATURE_COLS = [
    "top_wr_diff",
    "jg_wr_diff",
    "mid_wr_diff",
    "adc_wr_diff",
    "sup_wr_diff",
    "avg_wr_diff",
    "top_low_games_flag",
    "jg_low_games_flag",
    "mid_low_games_flag",
    "adc_low_games_flag",
    "sup_low_games_flag",
    "any_low_games_flag",
    "low_games_count",
]

# fixed model config
LOGREG_C = 1.0
LOGREG_PENALTY = "l2"


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


def take_subset(df: pd.DataFrame, n_rows: int | None) -> pd.DataFrame:
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


def build_pipeline() -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURE_COLS),
            ("num", numeric_transformer, NUMERIC_FEATURE_COLS),
        ]
    )

    model = LogisticRegression(
        C=LOGREG_C,
        penalty=LOGREG_PENALTY,
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
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


def collect_probs_and_labels(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    probs = pipeline.predict_proba(x)[:, 1]
    labels = y.to_numpy(dtype=float)
    return probs, labels


def make_calibration_table(
    probs: np.ndarray,
    labels: np.ndarray,
    bin_width: float = 0.05,
) -> pd.DataFrame:
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length.")

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

    return table[["bucket", "count", "avg_pred_prob", "actual_win_rate"]]


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
    plt.plot(x, y, marker="o", label="Logistic regression")
    plt.xlabel("Average predicted probability")
    plt.ylabel("Actual win rate")
    plt.title("Calibration plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_calibration_report(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    output_csv_path: Path,
    output_plot_path: Path,
    bin_width: float = 0.05,
) -> None:
    probs, labels = collect_probs_and_labels(pipeline, x, y)
    calibration_df = make_calibration_table(probs, labels, bin_width=bin_width)

    print("\nCalibration table:")
    print(calibration_df.to_string(index=False))

    calibration_df.to_csv(output_csv_path, index=False)
    plot_calibration_table(calibration_df, output_plot_path)

    print(f"\nSaved calibration table to: {output_csv_path.resolve()}")
    print(f"Saved calibration plot to:  {output_plot_path.resolve()}")


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

    x_train, y_train = make_xy(train_df)
    x_val, y_val = make_xy(val_df)
    x_test, y_test = make_xy(test_df)

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    train_loss, train_acc = evaluate_split(pipeline, x_train, y_train)
    val_loss, val_acc = evaluate_split(pipeline, x_val, y_val)
    test_loss, test_acc = evaluate_split(pipeline, x_test, y_test)

    print("\n=== Logistic Regression ===")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

    calibration_csv_path = OUTPUT_DIR / "test_calibration_5pct.csv"
    calibration_plot_path = OUTPUT_DIR / "test_calibration_5pct.png"

    print_calibration_report(
        pipeline=pipeline,
        x=x_test,
        y=y_test,
        output_csv_path=calibration_csv_path,
        output_plot_path=calibration_plot_path,
        bin_width=0.05,
    )

    print_example_predictions(pipeline, test_df, num_examples=5)


if __name__ == "__main__":
    main()