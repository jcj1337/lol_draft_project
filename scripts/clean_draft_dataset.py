from pathlib import Path
import pandas as pd

INTERIM_DIR = Path("data/processed")
PROCESSED_DIR = Path("data/cleaned")

BLUE_CHAMP_COLS = ["blue_top", "blue_jg", "blue_mid", "blue_adc", "blue_sup"]
RED_CHAMP_COLS = ["red_top", "red_jg", "red_mid", "red_adc", "red_sup"]

BLUE_WR_COLS = ["blue_top_wr", "blue_jg_wr", "blue_mid_wr", "blue_adc_wr", "blue_sup_wr"]
RED_WR_COLS = ["red_top_wr", "red_jg_wr", "red_mid_wr", "red_adc_wr", "red_sup_wr"]

BLUE_GAMES_COLS = ["blue_top_games", "blue_jg_games", "blue_mid_games", "blue_adc_games", "blue_sup_games"]
RED_GAMES_COLS = ["red_top_games", "red_jg_games", "red_mid_games", "red_adc_games", "red_sup_games"]

REQUIRED_COLS = [
    "match_id",
    "patch",
    "blue_win",
    "blue_avg_wr",
    "red_avg_wr",
    "avg_wr_diff",
    *BLUE_CHAMP_COLS,
    *RED_CHAMP_COLS,
    *BLUE_WR_COLS,
    *RED_WR_COLS,
    *BLUE_GAMES_COLS,
    *RED_GAMES_COLS,
]

NUMERIC_COLS = [
    "blue_win",
    "blue_avg_wr",
    "red_avg_wr",
    "avg_wr_diff",
    *BLUE_WR_COLS,
    *RED_WR_COLS,
    *BLUE_GAMES_COLS,
    *RED_GAMES_COLS,
]


def find_latest_input_csv(data_dir: Path) -> Path:
    candidates = list(data_dir.glob("draft_dataset*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No matching draft dataset CSVs found in {data_dir.resolve()}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def make_output_path(input_csv: Path) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR / f"{input_csv.stem}_cleaned.csv"


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correct data types 
    """
    df = df.copy()

    # strip champ strings
    for col in BLUE_CHAMP_COLS + RED_CHAMP_COLS + ["match_id", "patch"]:
        df[col] = df[col].astype(str).str.strip()

    # numeric coercion
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # games should be ints
    for col in BLUE_GAMES_COLS + RED_GAMES_COLS:
        df[col] = df[col].astype("Int64")

    return df


def validate_input(df: pd.DataFrame) -> None:
    """
    Validation function probably useless i cant lie
    """
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if df[NUMERIC_COLS].isna().any().any():
        bad_cols = [col for col in NUMERIC_COLS if df[col].isna().any()]
        raise ValueError(f"Found non-numeric or missing values in numeric columns: {bad_cols}")

    if not df["blue_win"].isin([0, 1]).all():
        raise ValueError("blue_win must only contain 0/1 values.")

    for cols, side in [(BLUE_CHAMP_COLS, "blue"), (RED_CHAMP_COLS, "red")]:
        if df[cols].isna().any().any():
            raise ValueError(f"Found missing champion names on {side} side.")
        if (df[cols] == "").any().any():
            raise ValueError(f"Found blank champion names on {side} side.")

    # reasonable bounds
    wr_cols = BLUE_WR_COLS + RED_WR_COLS + ["blue_avg_wr", "red_avg_wr"]
    for col in wr_cols:
        if ((df[col] < 0) | (df[col] > 1)).any():
            raise ValueError(f"Win-rate column out of [0,1] range: {col}")

    games_cols = BLUE_GAMES_COLS + RED_GAMES_COLS
    for col in games_cols:
        if (df[col] < 0).any():
            raise ValueError(f"Games-played column has negative values: {col}")


def canonicalize_row(row: pd.Series) -> dict:
    blue_team = tuple(str(row[col]).strip() for col in BLUE_CHAMP_COLS)
    red_team = tuple(str(row[col]).strip() for col in RED_CHAMP_COLS)
    blue_win = int(row["blue_win"])

    blue_wr = tuple(float(row[col]) for col in BLUE_WR_COLS)
    red_wr = tuple(float(row[col]) for col in RED_WR_COLS)

    blue_games = tuple(int(row[col]) for col in BLUE_GAMES_COLS)
    red_games = tuple(int(row[col]) for col in RED_GAMES_COLS)

    blue_avg_wr = float(row["blue_avg_wr"])
    red_avg_wr = float(row["red_avg_wr"])

    if len(set(blue_team)) != 5:
        raise ValueError(f"Blue team does not have 5 unique champs in match {row['match_id']}")
    if len(set(red_team)) != 5:
        raise ValueError(f"Red team does not have 5 unique champs in match {row['match_id']}")
    if set(blue_team) & set(red_team):
        raise ValueError(f"Champion overlap between teams in match {row['match_id']}")

    if blue_team <= red_team:
        team_a = blue_team
        team_b = red_team
        team_a_wr = blue_wr
        team_b_wr = red_wr
        team_a_games = blue_games
        team_b_games = red_games
        team_a_avg_wr = blue_avg_wr
        team_b_avg_wr = red_avg_wr
        team_a_win = blue_win
    else:
        team_a = red_team
        team_b = blue_team
        team_a_wr = red_wr
        team_b_wr = blue_wr
        team_a_games = red_games
        team_b_games = blue_games
        team_a_avg_wr = red_avg_wr
        team_b_avg_wr = blue_avg_wr
        team_a_win = 1 - blue_win

    # existing raw WR diffs
    top_wr_diff = team_a_wr[0] - team_b_wr[0]
    jg_wr_diff = team_a_wr[1] - team_b_wr[1]
    mid_wr_diff = team_a_wr[2] - team_b_wr[2]
    adc_wr_diff = team_a_wr[3] - team_b_wr[3]
    sup_wr_diff = team_a_wr[4] - team_b_wr[4]

    # games summaries
    team_a_avg_games = sum(team_a_games) / 5
    team_b_avg_games = sum(team_b_games) / 5
    avg_games_diff = team_a_avg_games - team_b_avg_games

    # role-wise games diffs
    top_games_diff = team_a_games[0] - team_b_games[0]
    jg_games_diff = team_a_games[1] - team_b_games[1]
    mid_games_diff = team_a_games[2] - team_b_games[2]
    adc_games_diff = team_a_games[3] - team_b_games[3]
    sup_games_diff = team_a_games[4] - team_b_games[4]

    # role-wise min/max games
    top_min_games = min(team_a_games[0], team_b_games[0])
    jg_min_games = min(team_a_games[1], team_b_games[1])
    mid_min_games = min(team_a_games[2], team_b_games[2])
    adc_min_games = min(team_a_games[3], team_b_games[3])
    sup_min_games = min(team_a_games[4], team_b_games[4])

    top_max_games = max(team_a_games[0], team_b_games[0])
    jg_max_games = max(team_a_games[1], team_b_games[1])
    mid_max_games = max(team_a_games[2], team_b_games[2])
    adc_max_games = max(team_a_games[3], team_b_games[3])
    sup_max_games = max(team_a_games[4], team_b_games[4])

    # low-games flags
    LOW_GAMES_THRESHOLD = 20

    team_a_top_low_games = int(team_a_games[0] < LOW_GAMES_THRESHOLD)
    team_a_jg_low_games = int(team_a_games[1] < LOW_GAMES_THRESHOLD)
    team_a_mid_low_games = int(team_a_games[2] < LOW_GAMES_THRESHOLD)
    team_a_adc_low_games = int(team_a_games[3] < LOW_GAMES_THRESHOLD)
    team_a_sup_low_games = int(team_a_games[4] < LOW_GAMES_THRESHOLD)

    team_b_top_low_games = int(team_b_games[0] < LOW_GAMES_THRESHOLD)
    team_b_jg_low_games = int(team_b_games[1] < LOW_GAMES_THRESHOLD)
    team_b_mid_low_games = int(team_b_games[2] < LOW_GAMES_THRESHOLD)
    team_b_adc_low_games = int(team_b_games[3] < LOW_GAMES_THRESHOLD)
    team_b_sup_low_games = int(team_b_games[4] < LOW_GAMES_THRESHOLD)

    top_low_games_flag = int(top_min_games < LOW_GAMES_THRESHOLD)
    jg_low_games_flag = int(jg_min_games < LOW_GAMES_THRESHOLD)
    mid_low_games_flag = int(mid_min_games < LOW_GAMES_THRESHOLD)
    adc_low_games_flag = int(adc_min_games < LOW_GAMES_THRESHOLD)
    sup_low_games_flag = int(sup_min_games < LOW_GAMES_THRESHOLD)

    any_low_games_flag = int(
        top_low_games_flag
        or jg_low_games_flag
        or mid_low_games_flag
        or adc_low_games_flag
        or sup_low_games_flag
    )

    low_games_count = (
        team_a_top_low_games + team_a_jg_low_games + team_a_mid_low_games + team_a_adc_low_games + team_a_sup_low_games
        + team_b_top_low_games + team_b_jg_low_games + team_b_mid_low_games + team_b_adc_low_games + team_b_sup_low_games
    )

    min_games_in_match = min(*team_a_games, *team_b_games)
    max_games_in_match = max(*team_a_games, *team_b_games)

    # smoothed WRs
    # shrink low-sample WRs toward 0.5
    SMOOTHING_ALPHA = 20.0

    def smooth_wr(wr: float, games: int, alpha: float = SMOOTHING_ALPHA) -> float:
        approx_wins = wr * games
        return (approx_wins + alpha * 0.5) / (games + alpha)

    team_a_top_smoothed_wr = smooth_wr(team_a_wr[0], team_a_games[0])
    team_a_jg_smoothed_wr = smooth_wr(team_a_wr[1], team_a_games[1])
    team_a_mid_smoothed_wr = smooth_wr(team_a_wr[2], team_a_games[2])
    team_a_adc_smoothed_wr = smooth_wr(team_a_wr[3], team_a_games[3])
    team_a_sup_smoothed_wr = smooth_wr(team_a_wr[4], team_a_games[4])

    team_b_top_smoothed_wr = smooth_wr(team_b_wr[0], team_b_games[0])
    team_b_jg_smoothed_wr = smooth_wr(team_b_wr[1], team_b_games[1])
    team_b_mid_smoothed_wr = smooth_wr(team_b_wr[2], team_b_games[2])
    team_b_adc_smoothed_wr = smooth_wr(team_b_wr[3], team_b_games[3])
    team_b_sup_smoothed_wr = smooth_wr(team_b_wr[4], team_b_games[4])

    top_smoothed_wr_diff = team_a_top_smoothed_wr - team_b_top_smoothed_wr
    jg_smoothed_wr_diff = team_a_jg_smoothed_wr - team_b_jg_smoothed_wr
    mid_smoothed_wr_diff = team_a_mid_smoothed_wr - team_b_mid_smoothed_wr
    adc_smoothed_wr_diff = team_a_adc_smoothed_wr - team_b_adc_smoothed_wr
    sup_smoothed_wr_diff = team_a_sup_smoothed_wr - team_b_sup_smoothed_wr

    team_a_avg_smoothed_wr = (
        team_a_top_smoothed_wr
        + team_a_jg_smoothed_wr
        + team_a_mid_smoothed_wr
        + team_a_adc_smoothed_wr
        + team_a_sup_smoothed_wr
    ) / 5

    team_b_avg_smoothed_wr = (
        team_b_top_smoothed_wr
        + team_b_jg_smoothed_wr
        + team_b_mid_smoothed_wr
        + team_b_adc_smoothed_wr
        + team_b_sup_smoothed_wr
    ) / 5

    avg_smoothed_wr_diff = team_a_avg_smoothed_wr - team_b_avg_smoothed_wr

    return {
        "match_id": row["match_id"],

        "patch": row["patch"],

        "team_a_top": team_a[0],
        "team_a_jg": team_a[1],
        "team_a_mid": team_a[2],
        "team_a_adc": team_a[3],
        "team_a_sup": team_a[4],

        "team_b_top": team_b[0],
        "team_b_jg": team_b[1],
        "team_b_mid": team_b[2],
        "team_b_adc": team_b[3],
        "team_b_sup": team_b[4],

        # raw WRs
        "team_a_top_wr": team_a_wr[0],
        "team_a_jg_wr": team_a_wr[1],
        "team_a_mid_wr": team_a_wr[2],
        "team_a_adc_wr": team_a_wr[3],
        "team_a_sup_wr": team_a_wr[4],

        "team_b_top_wr": team_b_wr[0],
        "team_b_jg_wr": team_b_wr[1],
        "team_b_mid_wr": team_b_wr[2],
        "team_b_adc_wr": team_b_wr[3],
        "team_b_sup_wr": team_b_wr[4],

        # raw WR diffs
        "top_wr_diff": top_wr_diff,
        "jg_wr_diff": jg_wr_diff,
        "mid_wr_diff": mid_wr_diff,
        "adc_wr_diff": adc_wr_diff,
        "sup_wr_diff": sup_wr_diff,

        # games played
        "team_a_top_games": team_a_games[0],
        "team_a_jg_games": team_a_games[1],
        "team_a_mid_games": team_a_games[2],
        "team_a_adc_games": team_a_games[3],
        "team_a_sup_games": team_a_games[4],

        "team_b_top_games": team_b_games[0],
        "team_b_jg_games": team_b_games[1],
        "team_b_mid_games": team_b_games[2],
        "team_b_adc_games": team_b_games[3],
        "team_b_sup_games": team_b_games[4],

        "team_a_avg_games": team_a_avg_games,
        "team_b_avg_games": team_b_avg_games,
        "avg_games_diff": avg_games_diff,

        # new role-wise games features
        "top_games_diff": top_games_diff,
        "jg_games_diff": jg_games_diff,
        "mid_games_diff": mid_games_diff,
        "adc_games_diff": adc_games_diff,
        "sup_games_diff": sup_games_diff,

        "top_min_games": top_min_games,
        "jg_min_games": jg_min_games,
        "mid_min_games": mid_min_games,
        "adc_min_games": adc_min_games,
        "sup_min_games": sup_min_games,

        "top_max_games": top_max_games,
        "jg_max_games": jg_max_games,
        "mid_max_games": mid_max_games,
        "adc_max_games": adc_max_games,
        "sup_max_games": sup_max_games,

        # low-games flags
        "team_a_top_low_games": team_a_top_low_games,
        "team_a_jg_low_games": team_a_jg_low_games,
        "team_a_mid_low_games": team_a_mid_low_games,
        "team_a_adc_low_games": team_a_adc_low_games,
        "team_a_sup_low_games": team_a_sup_low_games,

        "team_b_top_low_games": team_b_top_low_games,
        "team_b_jg_low_games": team_b_jg_low_games,
        "team_b_mid_low_games": team_b_mid_low_games,
        "team_b_adc_low_games": team_b_adc_low_games,
        "team_b_sup_low_games": team_b_sup_low_games,

        "top_low_games_flag": top_low_games_flag,
        "jg_low_games_flag": jg_low_games_flag,
        "mid_low_games_flag": mid_low_games_flag,
        "adc_low_games_flag": adc_low_games_flag,
        "sup_low_games_flag": sup_low_games_flag,

        "any_low_games_flag": any_low_games_flag,
        "low_games_count": low_games_count,
        "min_games_in_match": min_games_in_match,
        "max_games_in_match": max_games_in_match,

        # smoothed WRs
        "team_a_top_smoothed_wr": team_a_top_smoothed_wr,
        "team_a_jg_smoothed_wr": team_a_jg_smoothed_wr,
        "team_a_mid_smoothed_wr": team_a_mid_smoothed_wr,
        "team_a_adc_smoothed_wr": team_a_adc_smoothed_wr,
        "team_a_sup_smoothed_wr": team_a_sup_smoothed_wr,

        "team_b_top_smoothed_wr": team_b_top_smoothed_wr,
        "team_b_jg_smoothed_wr": team_b_jg_smoothed_wr,
        "team_b_mid_smoothed_wr": team_b_mid_smoothed_wr,
        "team_b_adc_smoothed_wr": team_b_adc_smoothed_wr,
        "team_b_sup_smoothed_wr": team_b_sup_smoothed_wr,

        "top_smoothed_wr_diff": top_smoothed_wr_diff,
        "jg_smoothed_wr_diff": jg_smoothed_wr_diff,
        "mid_smoothed_wr_diff": mid_smoothed_wr_diff,
        "adc_smoothed_wr_diff": adc_smoothed_wr_diff,
        "sup_smoothed_wr_diff": sup_smoothed_wr_diff,

        "team_a_avg_wr": team_a_avg_wr,
        "team_b_avg_wr": team_b_avg_wr,
        "avg_wr_diff": team_a_avg_wr - team_b_avg_wr,

        "team_a_avg_smoothed_wr": team_a_avg_smoothed_wr,
        "team_b_avg_smoothed_wr": team_b_avg_smoothed_wr,
        "avg_smoothed_wr_diff": avg_smoothed_wr_diff,

        "team_a_win": int(team_a_win),
    }


def main() -> None:
    input_csv = find_latest_input_csv(INTERIM_DIR)
    output_csv = make_output_path(input_csv)

    df = pd.read_csv(input_csv)
    validate_input(df)
    df = coerce_types(df)
    validate_input(df)

    df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    rows = [canonicalize_row(row) for _, row in df.iterrows()]
    out_df = pd.DataFrame(rows)

    out_df.to_csv(output_csv, index=False)

    print(f"Input file:  {input_csv.resolve()}")
    print(f"Output file: {output_csv.resolve()}")
    print(f"Saved {len(out_df)} cleaned rows")
    print(out_df.head())


if __name__ == "__main__":
    main()