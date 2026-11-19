"""
Draft-only features for linear win-probability models.

Every feature is antisymmetric: blue terms are +1 and red terms are -1. Swapping
the two teams negates the feature vector, so a logistic regression on these
features automatically satisfies P(blue wins) = 1 - P(red wins) apart from the
intercept, which absorbs the blue-side advantage. This replaces the canonical
team ordering used in clean_draft_dataset.py.

No player features (win rate, games played) are used anywhere in this module.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

ROLES = ["top", "jg", "mid", "adc", "sup"]
BLUE_COLS = [f"blue_{r}" for r in ROLES]
RED_COLS = [f"red_{r}" for r in ROLES]
CHAMP_COLS = BLUE_COLS + RED_COLS
LABEL_COL = "blue_win"

# head-to-head pairs between enemy champions: (blue role, red role)
MATCHUP_PAIRS = [
    ("top", "top"),
    ("jg", "jg"),
    ("mid", "mid"),
    ("adc", "adc"),
    ("sup", "sup"),
    ("adc", "sup"),  # adc vs enemy support (also covers support vs enemy adc)
]

# same-team pairs
SYNERGY_PAIRS = [
    ("adc", "sup"),
    ("jg", "mid"),
    ("jg", "top"),
    ("jg", "sup"),
]

FEATURE_GROUPS = ["champ", "champ_role", "matchup", "synergy"]
PAIR_GROUPS = {"matchup", "synergy"}

RANK_GROUPS = ("core", "mixed", "emerald")
RANK_GROUP_DTYPE = pd.CategoricalDtype(list(RANK_GROUPS))
CORE_MIN_DIAMOND = 8
NON_CORE_GROUP_OF_CRAWLER = {"diamond_plus": "mixed", "emerald": "emerald"}
UNTAGGED_CRAWLER = "diamond_plus"  # the collector schema's default for crawled_from
RANK_COLS = ["n_diamond_plus", "n_emerald", "crawled_from"]


# ------------------------------ loading ------------------------------
def load_raw_drafts(csv_path: str | Path, min_major_patch: int = 16) -> pd.DataFrame:
    """
    Load the raw (blue/red) dataset produced by build_draft_dataset.py or a collect_drafts.py
    export, keeping only draft and rank columns, plus `rank_group` (see RANK_GROUPS). Patch is
    read as a string: as a float, "16.1" and "16.10" collide. The ten champion columns share one
    categorical dtype, so slices of the frame and their concatenation stay categorical.
    """
    keep = {"match_id", "source_platform", "patch", "game_creation", *CHAMP_COLS, LABEL_COL, *RANK_COLS}
    dtype = {
        "patch": str,
        "source_platform": "category",
        "crawled_from": "category",
        "n_diamond_plus": "int8",
        "n_emerald": "Int8",
        **dict.fromkeys(CHAMP_COLS, "category"),
    }
    df = pd.read_csv(csv_path, usecols=lambda c: c in keep, dtype=dtype)
    if df[CHAMP_COLS].isna().any(axis=None):
        raise ValueError(f"{csv_path} has games with a missing champion")
    champions = sorted(set().union(*(df[c].cat.categories for c in CHAMP_COLS)))
    for c in CHAMP_COLS:
        # set_categories, not astype: astype skips columns whose categories are the same set in another order
        df[c] = df[c].cat.set_categories(champions)

    df["patch_major"] = df["patch"].str.split(".").str[0].astype(int)
    df["patch_minor"] = df["patch"].str.split(".").str[1].astype(int)
    if "game_creation" in df.columns:
        # collect_drafts.py exports have a real timestamp
        df["time_order"] = df["game_creation"].astype(np.int64)
    else:
        # older datasets: match ids increase over time within a platform, e.g. NA1_5516057846
        df["time_order"] = df["match_id"].str.split("_").str[1].astype(np.int64)

    df = df[df["patch_major"] >= min_major_patch]
    df = df.drop_duplicates("match_id").reset_index(drop=True)
    df["rank_group"] = _rank_group(df)
    return df


def _rank_group(df: pd.DataFrame) -> pd.Categorical:
    core = RANK_GROUPS.index("core")
    if "n_diamond_plus" not in df.columns:
        return pd.Categorical.from_codes(np.full(len(df), core, dtype=np.int8), dtype=RANK_GROUP_DTYPE)
    group = np.full(len(df), RANK_GROUPS.index(NON_CORE_GROUP_OF_CRAWLER[UNTAGGED_CRAWLER]), dtype=np.int8)
    if "crawled_from" in df.columns:
        crawled = df["crawled_from"].cat
        unknown = set(crawled.categories) - NON_CORE_GROUP_OF_CRAWLER.keys()
        if unknown:
            raise ValueError(f"crawled_from values {sorted(unknown)} have no rank group in NON_CORE_GROUP_OF_CRAWLER")
        group_of_crawler = np.array([RANK_GROUPS.index(NON_CORE_GROUP_OF_CRAWLER[c]) for c in crawled.categories], dtype=np.int8)
        codes = crawled.codes.to_numpy()
        tagged = codes >= 0
        group[tagged] = group_of_crawler[codes[tagged]]
    group[df["n_diamond_plus"].to_numpy() >= CORE_MIN_DIAMOND] = core
    return pd.Categorical.from_codes(group, dtype=RANK_GROUP_DTYPE)


def within_patch_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    test  = a random `test_frac` of each patch's games
    val   = a random `val_frac` of each patch's games
    train = the rest
    The live recommender is retrained on the newest games, so it usually has games from the
    current patch. Testing on an unseen patch would add a handicap unrelated to draft modelling,
    and inside one patch a random draw showed no measurable difference from a date cut.
    """
    u = pd.Series(np.random.default_rng(seed).random(len(df)), index=df.index)
    pct = u.groupby(df["patch"]).rank(pct=True)
    test_mask = pct <= test_frac
    val_mask = (pct > test_frac) & (pct <= test_frac + val_frac)
    train_mask = ~test_mask & ~val_mask
    return (
        df[train_mask].reset_index(drop=True),
        df[val_mask].reset_index(drop=True),
        df[test_mask].reset_index(drop=True),
    )


# ------------------------------ features ------------------------------
def row_teams(row) -> tuple[dict[str, str], dict[str, str]]:
    blue = {r: getattr(row, f"blue_{r}") for r in ROLES}
    red = {r: getattr(row, f"red_{r}") for r in ROLES}
    return blue, red


def _arity(group: str) -> int:
    return 2 if group in PAIR_GROUPS else 1


@dataclass(frozen=True)
class _Slot:
    block: tuple[str, ...]
    a: int
    b: int | None = None
    sign: float = 1.0
    unordered: bool = False


def _slots(groups: Sequence[str]) -> tuple[_Slot, ...]:
    blue = {r: CHAMP_COLS.index(f"blue_{r}") for r in ROLES}
    red = {r: CHAMP_COLS.index(f"red_{r}") for r in ROLES}
    slots: list[_Slot] = []
    if "champ" in groups:
        for r in ROLES:
            slots += [_Slot(("champ",), blue[r]), _Slot(("champ",), red[r], sign=-1.0)]
    if "champ_role" in groups:
        for r in ROLES:
            slots += [_Slot(("champ_role", r), blue[r]), _Slot(("champ_role", r), red[r], sign=-1.0)]
    if "matchup" in groups:
        for r1, r2 in MATCHUP_PAIRS:
            block = ("matchup", r1, r2)
            if r1 == r2:
                slots.append(_Slot(block, blue[r1], red[r1], unordered=True))
            else:
                slots += [_Slot(block, blue[r1], red[r2]), _Slot(block, red[r1], blue[r2], sign=-1.0)]
    if "synergy" in groups:
        for r1, r2 in SYNERGY_PAIRS:
            block = ("synergy", r1, r2)
            slots += [_Slot(block, blue[r1], blue[r2]), _Slot(block, red[r1], red[r2], sign=-1.0)]
    return tuple(slots)


class _KeySpace:
    def __init__(self, champions: Iterable[str], slots: Sequence[_Slot]):
        self.names = sorted(champions)
        self.index = pd.Index(self.names)
        self.code = {name: i for i, name in enumerate(self.names)}
        self.k = len(self.names)
        self.start: dict[tuple[str, ...], int] = {}
        size = 0
        for s in slots:
            if s.block not in self.start:
                self.start[s.block] = size
                size += self.k ** _arity(s.block[0])
        self.size = size

    def terms(self, codes: np.ndarray, slots: Sequence[_Slot], pair_scale: float) -> tuple[np.ndarray, np.ndarray]:
        ids = np.empty((len(codes), len(slots)), dtype=np.int64)
        values = np.empty((len(codes), len(slots)))
        for t, s in enumerate(slots):
            a = codes[:, s.a].astype(np.int64)
            if s.b is None:
                ids[:, t] = np.where(a < 0, -1, self.start[s.block] + a)
                values[:, t] = s.sign
                continue
            b = codes[:, s.b].astype(np.int64)
            first, second = (np.minimum(a, b), np.maximum(a, b)) if s.unordered else (a, b)
            ids[:, t] = np.where((a < 0) | (b < 0), -1, self.start[s.block] + first * self.k + second)
            sign = np.where(a <= b, s.sign, -s.sign) if s.unordered else s.sign
            values[:, t] = sign * pair_scale
        return ids, values

    def counts(self, totals: np.ndarray) -> Counter:
        out: Counter = Counter()
        for block, start in self.start.items():
            arity = _arity(block[0])
            block_totals = totals[start:start + self.k ** arity]
            seen = np.flatnonzero(block_totals)
            for i, n in zip(seen.tolist(), block_totals[seen].tolist()):
                names = (self.names[i],) if arity == 1 else (self.names[i // self.k], self.names[i % self.k])
                out[(*block, *names)] = n
        return out

    def id_of(self, key: tuple) -> int:
        arity = _arity(key[0])
        local = 0
        for name in key[-arity:]:
            local = local * self.k + self.code[name]
        return self.start[key[:-arity]] + local


CHUNK_ROWS = 250_000


def _chunks(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
    for start in range(0, len(df), CHUNK_ROWS):
        yield df.iloc[start:start + CHUNK_ROWS]


def _frame_champions(df: pd.DataFrame) -> set[str]:
    names: set = set()
    for c in CHAMP_COLS:
        col = df[c]
        names.update(col.cat.categories if isinstance(col.dtype, pd.CategoricalDtype) else col.dropna().unique())
    return names


def _frame_codes(df: pd.DataFrame, space: _KeySpace) -> np.ndarray:
    codes = np.empty((len(df), len(CHAMP_COLS)), dtype=np.int16)
    for j, c in enumerate(CHAMP_COLS):
        col = df[c]
        if col.isna().any():
            raise ValueError(f"missing champion in column {c}")
        if isinstance(col.dtype, pd.CategoricalDtype):
            codes[:, j] = space.index.get_indexer(col.cat.categories)[col.cat.codes.to_numpy()]
        else:
            codes[:, j] = space.index.get_indexer(col)
    return codes


def _draft_codes(drafts: Sequence[tuple[dict, dict]], space: _KeySpace) -> np.ndarray:
    codes = [[space.code.get(team.get(r), -1) for team in (blue, red) for r in ROLES] for blue, red in drafts]
    return np.array(codes, dtype=np.int16).reshape(len(drafts), len(CHAMP_COLS))


class DraftFeaturizer:
    """
    Builds a sparse design matrix from drafts. The vocabulary is fit on training rows;
    pair features seen fewer than `min_pair_count` times in training are dropped, and
    pair features are multiplied by `pair_scale` (a smaller scale means stronger
    effective L2 shrinkage for rare, noisy pair effects).
    """

    def __init__(self, groups: list[str], min_pair_count: int = 10, pair_scale: float = 1.0):
        unknown = set(groups) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"Unknown feature groups: {unknown}")
        self.groups = groups
        self.min_pair_count = min_pair_count
        self.pair_scale = pair_scale
        self.vocab: dict[tuple, int] = {}
        self.counts: Counter = Counter()
        self.n_rows = 0
        self._index: tuple[_KeySpace, np.ndarray] | None = None

    def fit(self, df: pd.DataFrame) -> "DraftFeaturizer":
        slots = _slots(self.groups)
        space = _KeySpace(_frame_champions(df), slots)
        totals = np.zeros(space.size, dtype=np.int64)
        for chunk in _chunks(df):
            ids, _ = space.terms(_frame_codes(chunk, space), slots, self.pair_scale)
            totals += np.bincount(ids.ravel(), minlength=space.size)
        counts = space.counts(totals)

        keys = [
            k for k, c in counts.items()
            if k[0] not in PAIR_GROUPS or c >= self.min_pair_count
        ]
        self.vocab = {k: i for i, k in enumerate(sorted(keys))}
        self.counts = counts
        self.n_rows = len(df)
        self._index = None
        return self

    def transform_drafts(self, drafts: list[tuple[dict[str, str | None], dict[str, str | None]]]) -> sparse.csr_matrix:
        """A None or absent champion (a pick not made yet) adds no terms: it counts as an average champion."""
        space, _ = self._lookup()
        return self._matrix([_draft_codes(drafts, space)], len(drafts))

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        space, _ = self._lookup()
        return self._matrix((_frame_codes(chunk, space) for chunk in _chunks(df)), len(df))

    def feature_names(self) -> list[tuple]:
        names = [None] * len(self.vocab)
        for k, i in self.vocab.items():
            names[i] = k
        return names

    def _lookup(self) -> tuple[_KeySpace, np.ndarray]:
        if self._index is None:
            champions = {name for key in self.vocab for name in key[-_arity(key[0]):]}
            space = _KeySpace(champions, _slots(self.groups))
            column = np.full(space.size, -1, dtype=np.int32)
            for key, j in self.vocab.items():
                column[space.id_of(key)] = j
            self._index = (space, column)
        return self._index

    def _matrix(self, code_chunks: Iterable[np.ndarray], n_rows: int) -> sparse.csr_matrix:
        """
        The CSR matrix scipy builds from (values, (rows, columns)) over all rows: each chunk goes through
        that constructor, which sorts every row's columns and sums duplicate keys (one champion on both
        teams), and is copied into buffers with room for every slot of every row.
        """
        space, column = self._lookup()
        slots = _slots(self.groups)
        n_cols = len(self.vocab)
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        indices = np.empty(n_rows * len(slots), dtype=np.int32)
        data = np.empty(n_rows * len(slots))
        row = nnz = 0
        for codes in code_chunks:
            ids, values = space.terms(codes, slots, self.pair_scale)
            has_term = ids >= 0
            cols = np.full(ids.shape, -1, dtype=np.int32)
            cols[has_term] = column[ids[has_term]]
            hit = cols >= 0
            block = sparse.csr_matrix((values[hit], (np.nonzero(hit)[0], cols[hit])), shape=(len(codes), n_cols))
            indices[nnz:nnz + block.nnz] = block.indices
            data[nnz:nnz + block.nnz] = block.data
            indptr[row + 1:row + len(codes) + 1] = nnz + block.indptr[1:]
            row += len(codes)
            nnz += block.nnz
        return sparse.csr_matrix((data[:nnz], indices[:nnz], indptr), shape=(n_rows, n_cols))

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "_index"}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._index = None
