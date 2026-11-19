"""
Prove that the loader, DraftFeaturizer and LogRegModel in the working tree give exactly the
results of a pinned legacy revision.

    python -m scripts.verify_featurizer --csv <csv> [--legacy-rev 0bb2a7a] [--rows N]

The legacy src/draft_features.py and src/draft_models.py are loaded from `git show <rev>:<path>`
into throwaway modules and run next to the working tree's modules on the same CSV. The script
prints what it checked and exits with status 1 at the first mismatch. --rows checks only the
first N games of the CSV (for exports too large to load twice).
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import subprocess
import sys
import tempfile
import time
import types
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import src.draft_features as new
from scripts.recommend import MIN_CANDIDATE_PICK_RATE
from src.draft_features import BLUE_COLS, LABEL_COL, RED_COLS, ROLES

ROOT = Path(__file__).resolve().parents[1]
CHAMP_COLS = BLUE_COLS + RED_COLS
PUBLIC_STATE = {"groups", "min_pair_count", "pair_scale", "vocab", "counts", "n_rows"}
GROUP_SETS = (
    ("champ",),
    ("champ_role",),
    ("champ_role", "matchup"),
    ("champ_role", "matchup", "synergy"),
    ("champ", "champ_role", "matchup", "synergy"),
)
N_MASK_DRAFTS = 30
HELD_OUT_MIN_ROWS = 1000
DOWNSTREAM_ROWS = 100_000
DOWNSTREAM_GROUPS = ["champ_role", "matchup", "synergy"]
SEED = 42


@dataclass(frozen=True)
class Case:
    groups: tuple[str, ...]
    min_pair_count: int
    pair_scale: float

    def __str__(self) -> str:
        return f"{'+'.join(self.groups)} mpc={self.min_pair_count} ps={self.pair_scale}"


CASES = tuple(Case(g, m, s) for g in GROUP_SETS for m in (1, 10, 50) for s in (0.25, 1.0))


def check(ok: bool, what: str) -> None:
    if not ok:
        print(f"FAIL: {what}")
        sys.exit(1)


@contextmanager
def modules_as(overrides: dict[str, types.ModuleType]):
    saved = {name: sys.modules.get(name) for name in overrides}
    sys.modules.update(overrides)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_legacy(rev: str, path: str, overrides: dict[str, types.ModuleType] | None = None) -> types.ModuleType:
    """The file at `rev` as a module named like the real one, so its pickles name the real class path."""
    source = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    name = path.removesuffix(".py").replace("/", ".")
    module = types.ModuleType(name)
    module.__file__ = f"{rev}:{path}"
    with modules_as({**(overrides or {}), name: module}):
        exec(compile(source.decode("utf-8"), module.__file__, "exec"), module.__dict__)
    return module


def head_csv(csv: Path, rows: int, out_dir: Path) -> Path:
    out = out_dir / csv.name
    with open(csv, encoding="utf-8", newline="") as src, open(out, "w", encoding="utf-8", newline="") as dst:
        dst.writelines(itertools.islice(src, rows + 1))
    return out


def as_object(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({c: object for c in CHAMP_COLS})


def csr_difference(a: sparse.csr_matrix, b: sparse.csr_matrix) -> str | None:
    for what, x, y in (
        ("type", type(a), type(b)),
        ("shape", a.shape, b.shape),
        ("data dtype", a.data.dtype, b.data.dtype),
        ("indices dtype", a.indices.dtype, b.indices.dtype),
        ("indptr dtype", a.indptr.dtype, b.indptr.dtype),
    ):
        if x != y:
            return f"{what} {x} != {y}"
    for what in ("indptr", "indices", "data"):
        if not np.array_equal(getattr(a, what), getattr(b, what)):
            return f"{what} arrays differ"
    return None


def check_csr(a: sparse.csr_matrix, b: sparse.csr_matrix, what: str) -> None:
    diff = csr_difference(a, b)
    check(diff is None, f"{what}: {diff}")


def check_fit(old_feat, new_feat, what: str) -> None:
    check(list(old_feat.vocab.items()) == list(new_feat.vocab.items()), f"{what}: vocab (items in order)")
    check(type(new_feat.counts) is Counter, f"{what}: counts is a Counter")
    check(dict(old_feat.counts) == dict(new_feat.counts), f"{what}: counts as a dict")
    check(all(type(v) is int for v in new_feat.counts.values()), f"{what}: every count is a Python int")
    check(old_feat.n_rows == new_feat.n_rows, f"{what}: n_rows")
    check(old_feat.feature_names() == new_feat.feature_names(), f"{what}: feature_names()")


def check_frames(old_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
    check(len(old_df) == len(new_df), f"loader: {len(old_df)} legacy rows vs {len(new_df)}")
    kept = [c for c in new_df.columns if c in old_df.columns]
    check(kept == list(old_df.columns), f"loader: legacy columns {list(old_df.columns)} not kept in order ({kept})")
    for c in old_df.columns:
        same = np.array_equal(old_df[c].to_numpy(dtype=object), new_df[c].to_numpy(dtype=object))
        check(same, f"loader: values of column {c}")
    old_b, new_b = (d.memory_usage(deep=True).sum() / max(len(d), 1) for d in (old_df, new_df))
    print(f"ok loader: {len(new_df):,} rows, {len(old_df.columns)} legacy columns equal by value "
          f"(new extra: {[c for c in new_df.columns if c not in old_df.columns]}); {old_b:.0f} -> {new_b:.0f} B/row")


def check_split(old_split, new_split) -> None:
    for name, o, n in zip(("train", "val", "test"), old_split, new_split):
        check(o["match_id"].tolist() == n["match_id"].tolist(), f"split: {name} membership and order")
    print("ok split: within_patch_split(seed=42) " + " / ".join(f"{len(s):,}" for s in new_split) + " games, same match_ids in order")


def teams(df: pd.DataFrame, i: int) -> tuple[dict, dict]:
    row = df.iloc[i]
    return {r: row[f"blue_{r}"] for r in ROLES}, {r: row[f"red_{r}"] for r in ROLES}


def mask_drafts(df: pd.DataFrame) -> list[tuple[dict, dict]]:
    out = []
    for i in range(min(N_MASK_DRAFTS, len(df))):
        blue, red = teams(df, i)
        for mask in itertools.product((False, True), repeat=10):
            out.append((
                {r: None if mask[j] else blue[r] for j, r in enumerate(ROLES)},
                {r: None if mask[5 + j] else red[r] for j, r in enumerate(ROLES)},
            ))
    return out


def crafted_drafts(df: pd.DataFrame) -> list[tuple[dict, dict]]:
    blue, red = teams(df, 0)
    x = blue["top"]
    return [
        ({}, {}),
        ({r: None for r in ROLES}, {r: None for r in ROLES}),
        ({r: c for r, c in blue.items() if r != "mid"}, {r: c for r, c in red.items() if r not in ("top", "sup")}),
        ({**blue, "jg": "NotAChampion"}, {**red, "adc": "NotAChampion", "sup": "AlsoNotAChampion"}),
        (blue, {**red, "top": x}),
        ({**blue, "mid": x}, red),
        ({r: x for r in ROLES}, {r: x for r in ROLES}),
        ({"top": x}, {"top": x}),
    ]


def rank_drafts(feat, df: pd.DataFrame, rng: np.random.Generator) -> list[list[tuple[dict, dict]]]:
    role_games = {k[1:]: c for k, c in feat.counts.items() if k[0] == "champ_role"}
    batches = []
    for i in range(3):
        blue, red = teams(df, i)
        blue = {r: None if rng.random() < 0.3 else c for r, c in blue.items()}
        red = {r: None if rng.random() < 0.3 else c for r, c in red.items()}
        for side, role in itertools.product(("blue", "red"), ROLES):
            b0, r0 = dict(blue), dict(red)
            (b0 if side == "blue" else r0)[role] = None
            unavailable = {c for c in [*b0.values(), *r0.values()] if c}
            candidates = sorted(
                champ for (r, champ), n in role_games.items()
                if r == role and n / feat.n_rows >= MIN_CANDIDATE_PICK_RATE and champ not in unavailable
            )
            batch = []
            for champ in candidates:
                b, rd = dict(b0), dict(r0)
                (b if side == "blue" else rd)[role] = champ
                batch.append((b, rd))
            batches.append(batch)
    return batches


def check_transform_drafts(old_feat, new_feat, df: pd.DataFrame, what: str) -> int:
    rng = np.random.default_rng(SEED)
    batches = [mask_drafts(df), crafted_drafts(df), [], *rank_drafts(new_feat, df, rng)]
    for k, drafts in enumerate(batches):
        check_csr(old_feat.transform_drafts(drafts), new_feat.transform_drafts(drafts), f"{what}: transform_drafts batch {k}")
    return sum(len(b) for b in batches)


def check_case(legacy, case: Case, old_split, new_split, obj_split) -> None:
    args = (list(case.groups), case.min_pair_count, case.pair_scale)
    t = time.perf_counter()
    old_feat = legacy.DraftFeaturizer(*args).fit(old_split[0])
    t_old_fit = time.perf_counter() - t
    t = time.perf_counter()
    new_feat = new.DraftFeaturizer(*args).fit(new_split[0])
    t_new_fit = time.perf_counter() - t
    check_fit(old_feat, new_feat, f"{case}: fit on the loaded frame")
    check_fit(old_feat, new.DraftFeaturizer(*args).fit(old_split[0]), f"{case}: fit on the legacy frame")
    check_fit(old_feat, new.DraftFeaturizer(*args).fit(obj_split[0]), f"{case}: fit on an object frame")

    t_old_tr = t_new_tr = 0.0
    for name, o, n, b in zip(("train", "val", "test"), old_split, new_split, obj_split):
        t = time.perf_counter()
        expected = old_feat.transform(o)
        t_old_tr += time.perf_counter() - t
        t = time.perf_counter()
        got = new_feat.transform(n)
        t_new_tr += time.perf_counter() - t
        check_csr(expected, got, f"{case}: transform({name}) of the loaded frame")
        check_csr(expected, new_feat.transform(o), f"{case}: transform({name}) of the legacy frame")
        check_csr(expected, new_feat.transform(b), f"{case}: transform({name}) of an object frame")
        del expected, got
    n_drafts = check_transform_drafts(old_feat, new_feat, new_split[2], str(case))
    print(f"ok {case}: {len(new_feat.vocab):,} columns, {len(new_feat.counts):,} keys; "
          f"train/val/test equal from 3 frame types; {n_drafts:,} partial drafts; "
          f"legacy fit {t_old_fit:.1f}s transform {t_old_tr:.1f}s | new fit {t_new_fit:.2f}s transform {t_new_tr:.2f}s")


def check_held_out(legacy, old_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
    sizes = old_df["patch"].value_counts()
    if len(sizes) > 1:
        patch = sizes[sizes >= HELD_OUT_MIN_ROWS].idxmin()
        fit_mask = (old_df["patch"] == patch).to_numpy()
    else:
        patch = f"{sizes.index[0]}, first {HELD_OUT_MIN_ROWS:,} games"
        fit_mask = np.arange(len(old_df)) < HELD_OUT_MIN_ROWS
    rarest = pd.Series(old_df.loc[fit_mask, CHAMP_COLS].to_numpy().ravel()).value_counts().index[-1]
    fit_mask = fit_mask & ~old_df[CHAMP_COLS].eq(rarest).any(axis=1).to_numpy()
    rest_mask = ~fit_mask
    old_fit, old_rest = old_df[fit_mask].reset_index(drop=True), old_df[rest_mask].reset_index(drop=True)
    new_fit, new_rest = new_df[fit_mask].reset_index(drop=True), new_df[rest_mask].reset_index(drop=True)
    seen = set(old_fit[CHAMP_COLS].to_numpy().ravel())
    unseen_rows = int((~old_rest[CHAMP_COLS].isin(seen)).any(axis=1).sum())
    unseen_pairs = 0
    for groups in GROUP_SETS:
        what = f"held-out {'+'.join(groups)}"
        old_feat = legacy.DraftFeaturizer(list(groups), 10, 0.25).fit(old_fit)
        new_feat = new.DraftFeaturizer(list(groups), 10, 0.25).fit(new_fit)
        check_fit(old_feat, new_feat, f"{what}: fit on patch {patch}")
        expected = old_feat.transform(old_rest)
        check_csr(expected, new_feat.transform(new_rest), f"{what}: transform")
        check_csr(expected, new_feat.transform(as_object(new_rest)), f"{what}: transform of an object frame")
        if "synergy" in groups:
            rest_keys = new.DraftFeaturizer(list(groups), 1).fit(new_rest).counts
            unseen_pairs = sum(1 for k in rest_keys if k[0] in new.PAIR_GROUPS and k not in old_feat.counts)
    check(unseen_rows > 0 and unseen_pairs > 0, "held-out: the transformed games hold no unseen champion or pair")
    print(f"ok held-out: fit on patch {patch} without {rarest} ({len(old_fit):,} games), transform the other {len(old_rest):,} "
          f"({unseen_rows:,} hold a champion unseen in the fit, {unseen_pairs:,} pair keys unseen) for {len(GROUP_SETS)} group sets")


def check_repeated_champions(legacy, old_df: pd.DataFrame) -> None:
    frame = old_df.iloc[:400].reset_index(drop=True)
    frame.loc[::2, "red_top"] = frame.loc[::2, "blue_top"]
    frame.loc[::3, "blue_mid"] = frame.loc[::3, "blue_top"]
    frame.loc[::5, "red_sup"] = frame.loc[::5, "red_adc"]
    categorical = frame.astype(dict.fromkeys(CHAMP_COLS, "category"))
    for groups in GROUP_SETS:
        what = f"repeated champions {'+'.join(groups)}"
        old_feat = legacy.DraftFeaturizer(list(groups), 1, 0.25).fit(frame)
        for name, df in (("legacy", frame), ("categorical", categorical)):
            new_feat = new.DraftFeaturizer(list(groups), 1, 0.25).fit(df)
            check_fit(old_feat, new_feat, f"{what}: fit on the {name} frame")
            check_csr(old_feat.transform(frame), new_feat.transform(df), f"{what}: transform of the {name} frame")
    print(f"ok repeated champions: fit and transform {len(frame)} games with one champion in two slots, {len(GROUP_SETS)} group sets")


def check_pickles(legacy, old_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
    groups = list(GROUP_SETS[-1])
    old_feat = legacy.DraftFeaturizer(groups, 10, 0.5).fit(old_df)
    new_feat = new.DraftFeaturizer(groups, 10, 0.5).fit(new_df)
    drafts = mask_drafts(new_df)[:2048] + crafted_drafts(new_df)
    expected = old_feat.transform_drafts(drafts)
    new_feat.transform_drafts(drafts)

    with modules_as({"src.draft_features": legacy}):
        legacy_blob = pickle.dumps(old_feat)
    loaded = pickle.loads(legacy_blob)
    check(type(loaded) is new.DraftFeaturizer, "pickles: a legacy pickle loads as the new class")
    check_fit(old_feat, loaded, "pickles: legacy pickle loaded by the new class")
    check_csr(expected, loaded.transform_drafts(drafts), "pickles: legacy pickle, new class transform_drafts")

    new_blob = pickle.dumps(new_feat)
    with modules_as({"src.draft_features": legacy}):
        old_loaded = pickle.loads(new_blob)
    check(type(old_loaded) is legacy.DraftFeaturizer, "pickles: a new pickle loads as the legacy class")
    check(set(vars(old_loaded)) == PUBLIC_STATE, f"pickles: new pickle state is {sorted(vars(old_loaded))}")
    check_csr(expected, old_loaded.transform_drafts(drafts), "pickles: new pickle, legacy class transform_drafts")
    check_csr(expected, pickle.loads(new_blob).transform_drafts(drafts), "pickles: new pickle, new class transform_drafts")
    print(f"ok pickles: legacy -> new and new -> legacy both load; state keys {sorted(PUBLIC_STATE)}")


def check_downstream(rev: str, legacy, old_split, new_split) -> None:
    import torch

    import src.draft_models as new_models

    legacy_models = load_legacy(rev, "src/draft_models.py", {"src.draft_features": legacy})
    old_tr = old_split[0].iloc[:DOWNSTREAM_ROWS].reset_index(drop=True)
    new_tr = new_split[0].iloc[:DOWNSTREAM_ROWS].reset_index(drop=True)
    _, old_val, old_te = old_split
    _, new_val, new_te = new_split
    newest = max(new_tr["patch"].unique(), key=lambda p: tuple(map(int, p.split("."))))
    w = np.where(new_tr["patch"] == newest, 1.0, 0.5)

    t = time.perf_counter()
    old_model = legacy_models.LogRegModel(DOWNSTREAM_GROUPS)
    old_cfg = old_model.tune(old_tr, old_val, w)
    t_old = time.perf_counter() - t
    t = time.perf_counter()
    new_model = new_models.LogRegModel(DOWNSTREAM_GROUPS)
    new_cfg = new_model.tune(new_tr, new_val, w)
    t_new = time.perf_counter() - t
    check(old_cfg == new_cfg, f"downstream: tune config {old_cfg} vs {new_cfg}")
    check(np.array_equal(old_model.predict(old_te), new_model.predict(new_te)), "downstream: tuned predict() bit-equal")

    w2 = np.concatenate([w, np.full(len(old_val), 0.75)])
    old_model.refit(pd.concat([old_tr, old_val], ignore_index=True), w2)
    new_model.refit(pd.concat([new_tr, new_val], ignore_index=True), w2)
    for attr in ("coef_", "intercept_"):
        check(np.array_equal(getattr(old_model.model, attr), getattr(new_model.model, attr)), f"downstream: refit {attr} bit-equal")
    check(np.array_equal(old_model.predict(old_te), new_model.predict(new_te)), "downstream: refit predict() bit-equal")

    old_vocab, new_vocab = legacy_models.ChampionVocab(old_tr), new_models.ChampionVocab(new_tr)
    check(old_vocab.to_id == new_vocab.to_id, "downstream: ChampionVocab.to_id")
    for a, b in zip(old_vocab.encode(old_te), new_vocab.encode(new_te)):
        check(torch.equal(a, b), "downstream: ChampionVocab.encode")
    print(f"ok downstream: LogRegModel{DOWNSTREAM_GROUPS} weighted tune on {len(new_tr):,} games -> {new_cfg} "
          f"(legacy {t_old:.0f}s, new {t_new:.0f}s); tuned and refit predict() on {len(new_te):,} test games, "
          f"coef_ and intercept_ bit-equal; ChampionVocab equal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--legacy-rev", default="0bb2a7a")
    parser.add_argument("--rows", type=int, help="check only the first N games of the CSV")
    args = parser.parse_args()
    t0 = time.perf_counter()

    legacy = load_legacy(args.legacy_rev, "src/draft_features.py")
    with tempfile.TemporaryDirectory() as tmp:
        csv = head_csv(args.csv, args.rows, Path(tmp)) if args.rows else args.csv
        old_df, new_df = legacy.load_raw_drafts(csv), new.load_raw_drafts(csv)
    print(f"legacy {args.legacy_rev} vs working tree on {args.csv}" + (f" (first {args.rows:,} rows)" if args.rows else ""))
    check_frames(old_df, new_df)
    check_held_out(legacy, old_df, new_df)
    check_repeated_champions(legacy, old_df)
    check_pickles(legacy, old_df.iloc[:20_000], new_df.iloc[:20_000])

    old_split, new_split = legacy.within_patch_split(old_df, seed=SEED), new.within_patch_split(new_df, seed=SEED)
    del old_df, new_df
    check_split(old_split, new_split)
    obj_split = tuple(as_object(s) for s in new_split)
    for case in CASES:
        check_case(legacy, case, old_split, new_split, obj_split)
    del obj_split
    check_downstream(args.legacy_rev, legacy, old_split, new_split)
    print(f"PASS: {len(CASES)} featurizer cases, held-out, pickles, downstream ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
