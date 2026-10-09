"""
Pick recommendations from a draft-only logistic regression.

1. Train and save a model (all games in the CSV; older patches down-weighted by --gamma):
    python -m scripts.recommend fit --csv data/collector/exports/<export>.csv

2. Rank the champions you could pick. Give the picks made so far as role=Champion; roles are
   top, jg, mid, adc, sup. Picks not made yet can be left out (treated as an average champion).
    python -m scripts.recommend rank --side red --role mid ^
        --blue top=Gnar jg=Fizz mid=Ahri adc=Samira sup=Karma ^
        --red top=Urgot jg=Naafiri adc=Ezreal sup=Bard --bans Zed Yasuo

Champion names use Riot's internal names (e.g. MonkeyKing = Wukong); close matches are suggested.
"""
from __future__ import annotations

import argparse
import difflib
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.draft_features import LABEL_COL, ROLES, load_raw_drafts
from src.draft_models import LogRegModel

DEFAULT_CSV = Path("data/processed/draft_dataset_multiregion_diamondplus_110000.csv")
DEFAULT_MODEL = Path("outputs/recommender/model.pkl")
MIN_CANDIDATE_PICK_RATE = 0.01  # same filter as train_draft_baseline.py
# display names that differ from Riot's internal champion names
ALIASES = {"wukong": "monkeyking", "nunu&willump": "nunu", "nunuwillump": "nunu", "renataglasc": "renata"}


def patch_key(patch: str) -> tuple[int, int]:
    major, minor = patch.split(".")
    return int(major), int(minor)


def cmd_fit(args: argparse.Namespace) -> None:
    df = load_raw_drafts(args.csv)
    patches = sorted(df["patch"].unique(), key=patch_key)
    distance = df["patch"].map({p: len(patches) - 1 - i for i, p in enumerate(patches)}).to_numpy()
    weights = args.gamma ** distance

    model = LogRegModel(args.groups.split(","), name="recommender")
    model.config = {"C": args.C, "pair_scale": args.pair_scale}
    model.refit(df, weights)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "csv": str(args.csv),
        "games": len(df),
        "patches": patches,
        "gamma": args.gamma,
        "groups": model.groups,
        "config": model.config,
        "blue_side_win_rate": float(df[LABEL_COL].mean()),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(args.model, "wb") as f:
        pickle.dump({"feat": model.feat, "model": model.model, "meta": meta}, f)
    print(f"Trained on {len(df)} games (patches {patches[0]}-{patches[-1]}, gamma {args.gamma}); saved to {args.model}")


def parse_picks(items: list[str] | None, known: dict[str, str]) -> dict[str, str | None]:
    picks: dict[str, str | None] = {r: None for r in ROLES}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"Expected role=Champion, got {item!r}")
        role, champ = item.split("=", 1)
        if role not in ROLES:
            raise SystemExit(f"Unknown role {role!r}; use one of {ROLES}")
        picks[role] = resolve_champion(champ, known)
    return picks


def resolve_champion(name: str, known: dict[str, str]) -> str:
    key = name.lower().replace(" ", "").replace("'", "").replace(".", "")
    key = ALIASES.get(key, key)
    if key in known:
        return known[key]
    close = difflib.get_close_matches(key, known.keys(), n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(known[c] for c in close)}?" if close else ""
    raise SystemExit(f"Unknown champion {name!r}.{hint}")


def cmd_rank(args: argparse.Namespace) -> None:
    with open(args.model, "rb") as f:
        saved = pickle.load(f)
    feat, model, meta = saved["feat"], saved["model"], saved["meta"]

    role_games = {k[1:]: c for k, c in feat.counts.items() if k[0] == "champ_role"}
    known = {c.lower(): c for (_, c) in role_games}
    blue = parse_picks(args.blue, known)
    red = parse_picks(args.red, known)
    mine = blue if args.side == "blue" else red
    if mine[args.role] is not None:
        print(f"Ignoring the given {args.side} {args.role} pick ({mine[args.role]}); ranking that slot.")
        mine[args.role] = None

    unavailable = {c for c in list(blue.values()) + list(red.values()) if c} | {resolve_champion(b, known) for b in args.bans or []}
    candidates = sorted(
        champ for (role, champ), n in role_games.items()
        if role == args.role and n / feat.n_rows >= MIN_CANDIDATE_PICK_RATE and champ not in unavailable
    )

    drafts = []
    for champ in candidates:
        b, r = dict(blue), dict(red)
        (b if args.side == "blue" else r)[args.role] = champ
        drafts.append((b, r))
    p_blue = model.predict_proba(feat.transform_drafts(drafts))[:, 1]
    p_mine = p_blue if args.side == "blue" else 1 - p_blue

    table = pd.DataFrame({
        "champion": candidates,
        "win_prob": p_mine,
        "games_in_role": [role_games[(args.role, c)] for c in candidates],
    }).sort_values("win_prob", ascending=False).reset_index(drop=True)
    table["vs_median_option"] = table["win_prob"] - table["win_prob"].median()
    table.index += 1

    shown = pd.concat([table.head(args.top), table.tail(3)]) if len(table) > args.top + 3 else table
    print(f"Model: {meta['groups']} trained {meta['trained_at']} on {meta['games']} games, patches {meta['patches'][0]}-{meta['patches'][-1]}")
    print(f"Ranking {len(table)} {args.role} options for {args.side} side "
          f"(champions with >= {MIN_CANDIDATE_PICK_RATE:.0%} pick rate in the role, not picked or banned):\n")
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}" if abs(v) < 0.2 else f"{v:.3f}"):
        print(shown.to_string())
    print("\nDifferences of a few percentage points are typical; this model knows nothing about the players.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fit", help="train and save the recommender model")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--groups", default="champ_role,matchup,synergy")
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--pair-scale", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.0, help="weight multiplier per patch of age (1.0 = no down-weighting)")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("rank", help="rank the champions you could pick")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--side", choices=["blue", "red"], required=True)
    p.add_argument("--role", choices=ROLES, required=True)
    p.add_argument("--blue", nargs="*", help="blue picks so far, e.g. top=Gnar jg=Fizz")
    p.add_argument("--red", nargs="*", help="red picks so far")
    p.add_argument("--bans", nargs="*", help="banned champions")
    p.add_argument("--top", type=int, default=10, help="how many of the best options to show")
    p.set_defaults(func=cmd_rank)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
