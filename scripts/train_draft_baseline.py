"""
Draft-only baseline: L2 logistic regressions on champion / matchup / synergy features,
evaluated on a time-based split (test = latest patch).

Run from the project root:
    python -m scripts.train_draft_baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from src.draft_eval import compute_metrics, paired_bootstrap_ci, per_row_log_loss
from src.draft_features import (
    LABEL_COL,
    ROLES,
    DraftFeaturizer,
    load_raw_drafts,
    row_teams,
    time_split,
)

# -----------------------------
# Config
# -----------------------------
RAW_CSV = Path("data/processed/draft_dataset_multiregion_diamondplus_110000.csv")
OUTPUT_DIR = Path("outputs/draft_baseline")

SEED = 42

# (name, feature groups); None = constant blue-side win rate
MODELS: list[tuple[str, list[str] | None]] = [
    ("constant", None),
    ("champ", ["champ"]),
    ("champ_role", ["champ_role"]),
    ("champ_role+matchup", ["champ_role", "matchup"]),
    ("champ_role+matchup+synergy", ["champ_role", "matchup", "synergy"]),
]
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
PAIR_SCALE_GRID = [0.25, 0.5, 1.0]
MIN_PAIR_COUNT = 10

# example rankings
N_EXAMPLE_DRAFTS = 5
N_SPREAD_DRAFTS = 300          # drafts per role used to summarize how much the pick matters
# champion must be picked in the role in at least this share of training games to be ranked;
# filters off-role picks whose win rate mostly reflects one-trick players, not the pick
MIN_CANDIDATE_PICK_RATE = 0.01

WR_COLS = [f"{side}_{r}_wr" for side in ("blue", "red") for r in ROLES]


# -----------------------------
# Models
# -----------------------------
def fit_logreg(X: sparse.csr_matrix, y: np.ndarray, C: float) -> LogisticRegression:
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X, y)
    return model


def tune_on_val(
    groups: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    y_train = train_df[LABEL_COL].to_numpy()
    y_val = val_df[LABEL_COL].to_numpy()

    has_pairs = any(g in ("matchup", "synergy") for g in groups)
    scale_grid = PAIR_SCALE_GRID if has_pairs else [1.0]

    trials = []
    for pair_scale in scale_grid:
        feat = DraftFeaturizer(groups, min_pair_count=MIN_PAIR_COUNT, pair_scale=pair_scale).fit(train_df)
        X_train, X_val = feat.transform(train_df), feat.transform(val_df)
        for C in C_GRID:
            model = fit_logreg(X_train, y_train, C)
            p_val = model.predict_proba(X_val)[:, 1]
            trials.append({
                "C": C,
                "pair_scale": pair_scale,
                "n_features": X_train.shape[1],
                **compute_metrics(y_val, p_val),
            })

    best = min(trials, key=lambda t: t["log_loss"])
    return best, trials


def fit_final(groups: list[str], fit_df: pd.DataFrame, C: float, pair_scale: float) -> tuple[DraftFeaturizer, LogisticRegression]:
    feat = DraftFeaturizer(groups, min_pair_count=MIN_PAIR_COUNT, pair_scale=pair_scale).fit(fit_df)
    model = fit_logreg(feat.transform(fit_df), fit_df[LABEL_COL].to_numpy(), C)
    return feat, model


# -----------------------------
# Leakage check
# -----------------------------
def wr_diff_matrix(df: pd.DataFrame, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    diffs = np.column_stack([df[f"blue_{r}_wr"] - df[f"red_{r}_wr"] for r in ROLES])
    if mean is None:
        mean, std = diffs.mean(0), diffs.std(0)
    return sparse.csr_matrix((diffs - mean) / std), mean, std


def run_leakage_check(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    C: float,
) -> dict[str, dict[str, float]]:
    """champ_role + current-season win-rate diffs. For comparison only; these features leak the label."""
    wr = pd.read_csv(RAW_CSV, usecols=["match_id", *WR_COLS])
    train_df, val_df, test_df = (d.merge(wr, on="match_id", how="left") for d in (train_df, val_df, test_df))

    out = {}
    for split_name, fit_df, eval_df in [
        ("val", train_df, val_df),
        ("test", pd.concat([train_df, val_df], ignore_index=True), test_df),
    ]:
        feat = DraftFeaturizer(["champ_role"]).fit(fit_df)
        wr_fit, mean, std = wr_diff_matrix(fit_df)
        wr_eval, _, _ = wr_diff_matrix(eval_df, mean, std)
        X_fit = sparse.hstack([feat.transform(fit_df), wr_fit]).tocsr()
        X_eval = sparse.hstack([feat.transform(eval_df), wr_eval]).tocsr()
        model = fit_logreg(X_fit, fit_df[LABEL_COL].to_numpy(), C)
        out[split_name] = compute_metrics(eval_df[LABEL_COL].to_numpy(), model.predict_proba(X_eval)[:, 1])
    return out


# -----------------------------
# Rankings
# -----------------------------
def rank_candidates(
    feat: DraftFeaturizer,
    model: LogisticRegression,
    blue: dict[str, str],
    red: dict[str, str],
    side: str,
    role: str,
) -> pd.DataFrame:
    """Win probability for `side` for every eligible champion in `role`, holding the other 9 picks fixed."""
    taken = (set(blue.values()) | set(red.values())) - {(blue if side == "blue" else red)[role]}
    candidates = sorted(
        k[2] for k in feat.vocab
        if k[0] == "champ_role" and k[1] == role
        and feat.counts[k] / feat.n_rows >= MIN_CANDIDATE_PICK_RATE and k[2] not in taken
    )

    drafts = []
    for champ in candidates:
        b, r = dict(blue), dict(red)
        (b if side == "blue" else r)[role] = champ
        drafts.append((b, r))

    p_blue = model.predict_proba(feat.transform_drafts(drafts))[:, 1]
    p_side = p_blue if side == "blue" else 1 - p_blue
    return (
        pd.DataFrame({"champion": candidates, "win_prob": p_side})
        .sort_values("win_prob", ascending=False)
        .reset_index(drop=True)
    )


def write_example_rankings(
    feat: DraftFeaturizer,
    model: LogisticRegression,
    model_name: str,
    test_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    lines = [f"Example rankings from model: {model_name}", "Red side picks the hidden role; the other 9 picks are fixed.", ""]

    example_idx = rng.choice(len(test_df), N_EXAMPLE_DRAFTS, replace=False)
    for n, i in enumerate(example_idx):
        row = next(test_df.iloc[[i]].itertuples(index=False))
        blue, red = row_teams(row)
        role = ROLES[n % len(ROLES)]
        ranking = rank_candidates(feat, model, blue, red, "red", role)
        actual = red[role]

        lines.append(f"--- {row.match_id} | red {role} | actual pick: {actual} | red won: {1 - int(getattr(row, LABEL_COL))}")
        lines.append("Blue: " + ", ".join(f"{r}={blue[r]}" for r in ROLES))
        lines.append("Red:  " + ", ".join(f"{r}={red[r] if r != role else '???'}" for r in ROLES))
        for _, c in ranking.head(5).iterrows():
            lines.append(f"  {c['champion']:<14} {c['win_prob']:.3f}")
        lines.append("  ...")
        for _, c in ranking.tail(3).iterrows():
            lines.append(f"  {c['champion']:<14} {c['win_prob']:.3f}")
        hit = ranking.index[ranking["champion"] == actual]
        if len(hit):
            lines.append(f"  actual pick {actual}: rank {hit[0] + 1}/{len(ranking)}, win_prob {ranking.loc[hit[0], 'win_prob']:.3f}")
        else:
            lines.append(f"  actual pick {actual}: not eligible (< {MIN_CANDIDATE_PICK_RATE:.0%} pick rate in role)")
        lines.append("")

    # how much does the pick move the probability, according to the model?
    spread_rows = []
    for role in ROLES:
        for i in rng.choice(len(test_df), N_SPREAD_DRAFTS, replace=False):
            row = next(test_df.iloc[[i]].itertuples(index=False))
            blue, red = row_teams(row)
            ranking = rank_candidates(feat, model, blue, red, "red", role)
            actual = ranking.loc[ranking["champion"] == red[role], "win_prob"]
            spread_rows.append({
                "role": role,
                "n_candidates": len(ranking),
                "best_minus_worst": ranking["win_prob"].iloc[0] - ranking["win_prob"].iloc[-1],
                "best_minus_actual": ranking["win_prob"].iloc[0] - actual.iloc[0] if len(actual) else np.nan,
                "std": ranking["win_prob"].std(),
            })
    spread = pd.DataFrame(spread_rows).groupby("role").median().reindex(ROLES)

    lines.append("Median over test drafts of how much the hidden pick moves red's win probability:")
    lines.append(spread.to_string(float_format=lambda v: f"{v:.4f}"))
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return spread


# -----------------------------
# Plots / tables
# -----------------------------
def plot_calibration(preds: dict[str, np.ndarray], y: np.ndarray, out_path: Path, n_bins: int = 10) -> None:
    plt.figure(figsize=(6, 6))
    lo, hi = 1.0, 0.0
    for name, p in preds.items():
        order = np.argsort(p)
        bins = np.array_split(order, n_bins)
        xs = [p[b].mean() for b in bins]
        ys = [y[b].mean() for b in bins]
        lo, hi = min(lo, *xs, *ys), max(hi, *xs, *ys)
        plt.plot(xs, ys, marker="o", label=name)
    pad = 0.01
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1, label="perfect")
    plt.xlabel("Mean predicted P(blue win), decile")
    plt.ylabel("Actual blue win rate")
    plt.title("Test calibration (equal-count deciles)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def champion_role_table(feat: DraftFeaturizer, model: LogisticRegression) -> pd.DataFrame:
    coef = model.coef_[0]
    rows = [
        {"role": k[1], "champion": k[2], "coef": coef[i], "train_games": feat.counts[k]}
        for k, i in feat.vocab.items() if k[0] == "champ_role"
    ]
    return pd.DataFrame(rows).sort_values(["role", "coef"], ascending=[True, False])


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    global RAW_CSV, OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=RAW_CSV, help="raw blue/red draft CSV (e.g. a collect_drafts export)")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output directory")
    args = parser.parse_args()
    RAW_CSV, OUTPUT_DIR = args.csv, args.out
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_drafts(RAW_CSV)
    train_df, val_df, test_df = time_split(df)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Loaded {len(df)} games from {RAW_CSV}")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:<5} {len(d):>6} games | patches {sorted(d['patch'].unique(), key=lambda s: tuple(map(int, s.split('.'))))} | blue WR {d[LABEL_COL].mean():.4f}")

    y_val = val_df[LABEL_COL].to_numpy()
    y_test = test_df[LABEL_COL].to_numpy()

    results = []
    all_trials = []
    best_configs = {}
    test_preds: dict[str, np.ndarray] = {}
    final_models: dict[str, tuple[DraftFeaturizer, LogisticRegression]] = {}

    for name, groups in MODELS:
        if groups is None:
            p_val = np.full(len(val_df), train_df[LABEL_COL].mean())
            p_test = np.full(len(test_df), trainval_df[LABEL_COL].mean())
            val_metrics = compute_metrics(y_val, p_val)
            best_configs[name] = {}
        else:
            best, trials = tune_on_val(groups, train_df, val_df)
            all_trials += [{"model": name, **t} for t in trials]
            val_metrics = {k: best[k] for k in ("log_loss", "brier", "auc", "accuracy", "ece_q10", "pred_std")}
            best_configs[name] = {"C": best["C"], "pair_scale": best["pair_scale"], "n_features_train": best["n_features"]}

            feat, model = fit_final(groups, trainval_df, best["C"], best["pair_scale"])
            final_models[name] = (feat, model)
            p_test = model.predict_proba(feat.transform(test_df))[:, 1]

        test_preds[name] = p_test
        results.append({"model": name, "split": "val", **val_metrics})
        results.append({"model": name, "split": "test", **compute_metrics(y_test, p_test)})
        print(f"[{name}] config={best_configs[name]} val_log_loss={val_metrics['log_loss']:.5f}")

    # paired bootstrap on test log loss
    losses = {name: per_row_log_loss(y_test, p) for name, p in test_preds.items()}
    comparisons = []
    for name in test_preds:
        if name == "constant":
            continue
        for ref in ("constant", "champ_role"):
            if ref == name:
                continue
            mean, lo, hi = paired_bootstrap_ci(losses[ref], losses[name], SEED)
            comparisons.append({"model": name, "vs": ref, "log_loss_improvement": mean, "ci95_low": lo, "ci95_high": hi})
    comparisons_df = pd.DataFrame(comparisons)

    # leakage check, using champ_role's tuned C (only the old dataset has win-rate columns)
    if set(WR_COLS) <= set(pd.read_csv(RAW_CSV, nrows=0).columns):
        leak = run_leakage_check(train_df, val_df, test_df, best_configs["champ_role"]["C"])
        for split_name, m in leak.items():
            results.append({"model": "LEAKY champ_role+current_wr", "split": split_name, **m})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    comparisons_df.to_csv(OUTPUT_DIR / "test_bootstrap_vs_baselines.csv", index=False)
    pd.DataFrame(all_trials).to_csv(OUTPUT_DIR / "val_grid.csv", index=False)
    (OUTPUT_DIR / "best_configs.json").write_text(json.dumps(best_configs, indent=2))

    plot_calibration(
        {k: v for k, v in test_preds.items() if k != "constant"},
        y_test,
        OUTPUT_DIR / "test_calibration_deciles.png",
    )

    # champion strengths and rankings from the best non-constant model on val
    val_rows = results_df[(results_df["split"] == "val") & results_df["model"].isin(final_models)]
    best_name = val_rows.sort_values("log_loss").iloc[0]["model"]
    champion_role_table(*final_models["champ_role"]).to_csv(OUTPUT_DIR / "champion_role_coefs.csv", index=False)
    spread = write_example_rankings(*final_models[best_name], best_name, test_df, OUTPUT_DIR / "example_rankings.txt")

    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.5f}"):
        print("\n=== Results ===")
        print(results_df.to_string(index=False))
        print("\n=== Test log-loss improvement (positive = better), paired bootstrap 95% CI ===")
        print(comparisons_df.to_string(index=False))
        print(f"\n=== How much the pick moves win probability ({best_name}, medians) ===")
        print(spread.to_string())
    print(f"\nSaved outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
