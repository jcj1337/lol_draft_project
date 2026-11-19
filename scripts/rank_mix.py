"""
Should lower-rank games join the Diamond+ model's training set, and at what weight?

    python -m scripts.rank_mix --csv data/collector/exports/drafts_all_<n>.csv
        [--out outputs/rank_mix] [--weights 0.1 0.25 0.5 1] [--gamma 1.0] [--seed 42]

Only core games (see RANK_GROUPS in src/draft_features.py) are split, within patch, into
train/val/test. The other rank groups only ever join training, through training_set() with a
RankWeights, the same way `recommend fit --rank-weights` uses them. The model is the
recommender's logistic regression on champion, matchup and synergy features.

1. Core only: tune C and pair_scale on core val, giving (C0, s0).
2. One rank group at a time, in RANK_GROUPS order (groups with no games are skipped): try each
   weight with the groups before it held at their chosen weights. Every arm re-tunes C around C0
   at s0, because weighted extra games weaken the L2 penalty. A group keeps its best weight if
   that arm beats the arm chosen so far on core val, otherwise 0.
3. Uniform (every group at 1, today's `recommend fit` default) is tuned the same way.
4. The chosen arm has the lowest core-val log loss of all. Only then are core only, chosen and
   uniform refit on core train+val plus their other games and scored once on core test, with a
   paired bootstrap 95% CI against core only.

The same command answers the question before the Emerald merge (mixed games only) and after it.
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.draft_eval import compute_metrics, paired_bootstrap_ci, per_row_log_loss
from src.draft_features import (
    CHAMP_COLS,
    LABEL_COL,
    NON_CORE_GROUPS,
    RankWeights,
    load_raw_drafts,
    training_set,
    within_patch_split,
)
from src.draft_models import ConstantModel, LogRegModel

GROUPS = ["champ_role", "matchup", "synergy"]
CORE_C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)
C_STEP = 3.0
MAX_C_STEPS = 2
DEFAULT_WEIGHTS = (0.1, 0.25, 0.5, 1.0)
DEFAULT_OUT = Path("outputs/rank_mix")
MODEL_COLS = [*CHAMP_COLS, LABEL_COL, "patch", "rank_group"]


@dataclass(frozen=True)
class Arm:
    weights: RankWeights
    fit_rows: int
    sum_weight: float
    C: float
    pair_scale: float
    c_at_edge: bool
    val_log_loss: float


def val_log_loss(model: LogRegModel, val: pd.DataFrame) -> float:
    return compute_metrics(val[LABEL_COL].to_numpy(), model.predict(val))["log_loss"]


def tune(fit_df: pd.DataFrame, w: np.ndarray, val: pd.DataFrame, c_grid: tuple[float, ...],
         pair_scale_grid: tuple[float, ...]) -> tuple[LogRegModel, float, bool]:
    model = LogRegModel(GROUPS, c_grid=c_grid, pair_scale_grid=pair_scale_grid)
    model.tune(fit_df, val, w)
    loss, at_edge = val_log_loss(model, val), model.c_at_edge
    step = 1 / C_STEP if model.config["C"] == c_grid[0] else C_STEP
    for _ in range(MAX_C_STEPS):
        if not at_edge:
            break
        wider = LogRegModel(GROUPS, c_grid=(model.config["C"] * step,), pair_scale_grid=pair_scale_grid)
        wider.tune(fit_df, val, w)
        wider_loss = val_log_loss(wider, val)
        if wider_loss < loss:
            model, loss = wider, wider_loss
        else:
            at_edge = False
    return model, loss, at_edge


class Arms:
    def __init__(self, train: pd.DataFrame, val: pd.DataFrame, extra: pd.DataFrame, gamma: float):
        self.train, self.val, self.extra, self.gamma = train, val, extra, gamma
        self.present = [g for g in NON_CORE_GROUPS if (extra["rank_group"] == g).any()]
        self.tuned: dict[tuple[float, ...], Arm] = {}

    def present_weights(self, weights: RankWeights) -> tuple[float, ...]:
        return tuple(getattr(weights, g) for g in self.present)

    def tune(self, weights: RankWeights, c_grid: tuple[float, ...], pair_scale_grid: tuple[float, ...]) -> Arm:
        key = self.present_weights(weights)
        if key in self.tuned:
            return dataclasses.replace(self.tuned[key], weights=weights)
        t = time.time()
        fit_df, w = training_set(pd.concat([self.train, self.extra], ignore_index=True), weights, self.gamma)
        model, loss, at_edge = tune(fit_df, w, self.val, c_grid, pair_scale_grid)
        arm = Arm(weights, len(fit_df), float(w.sum()), model.config["C"], model.config["pair_scale"], at_edge, loss)
        self.tuned[key] = arm
        print(f"  {str(weights):<26}{len(fit_df):>11,} rows  C {arm.C:<8.4g}ps {arm.pair_scale:<6g}"
              f"val {loss:.5f}{'  (C at grid edge)' if at_edge else ''}  {time.time() - t:.0f}s", flush=True)
        return arm


def search(arms: Arms, weights_grid: list[float]) -> tuple[Arm, Arm, Arm]:
    core_only = arms.tune(RankWeights.core_only(), CORE_C_GRID, LogRegModel(GROUPS).pair_scale_grid)
    around = (core_only.C / C_STEP, core_only.C, core_only.C * C_STEP)
    best = core_only
    for group in arms.present:
        tried = [
            arms.tune(dataclasses.replace(best.weights, **{group: w}), around, (core_only.pair_scale,))
            for w in weights_grid
        ]
        best = min([best, *tried], key=lambda a: a.val_log_loss)
    uniform = arms.tune(RankWeights.uniform(), around, (core_only.pair_scale,))
    chosen = min(arms.tuned.values(), key=lambda a: a.val_log_loss)
    return core_only, chosen, uniform


def score_on_test(roles: dict[str, Arm], arms: Arms, train: pd.DataFrame, val: pd.DataFrame, test_df: pd.DataFrame,
         extra: pd.DataFrame, gamma: float, seed: int) -> pd.DataFrame:
    pool = pd.concat([train, val, extra], ignore_index=True)
    y = test_df[LABEL_COL].to_numpy()
    constant = ConstantModel()
    constant.refit(pd.concat([train, val], ignore_index=True))
    constant_loss = per_row_log_loss(y, constant.predict(test_df)).mean()
    losses: dict[tuple[float, ...], np.ndarray] = {}
    rows = []
    for role, arm in roles.items():
        key = arms.present_weights(arm.weights)
        if key not in losses:
            t = time.time()
            fit_df, w = training_set(pool, arm.weights, gamma)
            model = LogRegModel(GROUPS)
            model.config = {"C": arm.C, "pair_scale": arm.pair_scale}
            model.refit(fit_df, w)
            losses[key] = per_row_log_loss(y, model.predict(test_df))
            print(f"  {role:<10}{str(arm.weights):<26}{len(fit_df):>11,} rows  test {losses[key].mean():.5f}  "
                  f"{time.time() - t:.0f}s", flush=True)
        base = losses[arms.present_weights(roles["core only"].weights)]
        mean, lo, hi = paired_bootstrap_ci(base, losses[key], seed) if role != "core only" else (np.nan,) * 3
        rows.append({
            "arm": role, "weights": str(arm.weights), "C": arm.C, "pair_scale": arm.pair_scale,
            "test_log_loss": losses[key].mean(), "vs_constant": constant_loss - losses[key].mean(),
            "vs_core_only": mean, "ci95_low": lo, "ci95_high": hi,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True, help="a collect_drafts export with every rank")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS), help="weights tried for each rank group")
    parser.add_argument("--gamma", type=float, default=1.0, help="weight multiplier per patch of age, as in recommend fit")
    parser.add_argument("--seed", type=int, default=42, help="seed of the core split and the bootstrap")
    args = parser.parse_args()
    t0 = time.time()

    df = load_raw_drafts(args.csv)[MODEL_COLS]
    core = (df["rank_group"] == "core").to_numpy()
    train, val, test_df = within_patch_split(df[core].reset_index(drop=True), seed=args.seed)
    extra = df[~core].reset_index(drop=True)
    del df
    groups = extra["rank_group"].value_counts(sort=False)
    print(f"{args.csv}: core train {len(train):,} | val {len(val):,} | test {len(test_df):,}; other games: "
          + ", ".join(f"{g} {n:,}" for g, n in groups.items() if g != "core"))

    print("\nArms, tuned on core val:")
    arms = Arms(train, val, extra, args.gamma)
    core_only, chosen, uniform = search(arms, args.weights)
    table = pd.DataFrame([{
        "weights": str(a.weights), "fit_rows": a.fit_rows, "sum_weight": a.sum_weight, "C": a.C,
        "pair_scale": a.pair_scale, "c_at_edge": a.c_at_edge, "val_log_loss": a.val_log_loss, "chosen": a is chosen,
    } for a in arms.tuned.values()])

    print("\nTest, once, on core test games:")
    roles = {"core only": core_only, "chosen": chosen, "uniform": uniform}
    results = score_on_test(roles, arms, train, val, test_df, extra, args.gamma, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "arms.csv", index=False)
    results.to_csv(args.out / "test.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.6g}"):
        print("\n=== Arms (core val) ===")
        print(table.to_string(index=False))
        print("\n=== Test (core test; vs_* = log-loss improvement, positive = better; CI = paired bootstrap 95%) ===")
        print(results.to_string(index=False))
    gamma = f" --gamma {args.gamma:g}" if args.gamma != 1.0 else ""
    print(f"\nProduction fit with the val-chosen arm:\n  python -m scripts.recommend fit --csv {args.csv} "
          f"--rank-weights {chosen.weights} --C {chosen.C:.6g} --pair-scale {chosen.pair_scale:g}{gamma}")
    print(f"\nSaved {args.out / 'arms.csv'} and {args.out / 'test.csv'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
