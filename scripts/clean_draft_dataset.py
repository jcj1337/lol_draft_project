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


def find_latest_input_csv(data_dir: Path) -> Path:
    candidates = list(data_dir.glob("draft_dataset*.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"No matching draft dataset CSVs found in {data_dir.resolve()}"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def make_output_path(input_csv: Path) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR / f"{input_csv.stem}_cleaned.csv"


def validate_input(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if not df["blue_win"].isin([0, 1]).all():
        raise ValueError("blue_win must only contain 0/1 values.")

    for cols, side in [(BLUE_CHAMP_COLS, "blue"), (RED_CHAMP_COLS, "red")]:
        if df[cols].isna().any().any():
            raise ValueError(f"Found missing champion names on {side} side.")

    for cols, side in [(BLUE_WR_COLS, "blue"), (RED_WR_COLS, "red")]:
        if df[cols].isna().any().any():
            raise ValueError(f"Found missing win-rate values on {side} side.")

    for cols, side in [(BLUE_GAMES_COLS, "blue"), (RED_GAMES_COLS, "red")]:
        if df[cols].isna().any().any():
            raise ValueError(f"Found missing games-played values on {side} side.")


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

        "team_a_avg_wr": team_a_avg_wr,
        "team_b_avg_wr": team_b_avg_wr,
        "avg_wr_diff": team_a_avg_wr - team_b_avg_wr,

        "team_a_win": int(team_a_win),
    }


def main() -> None:
    input_csv = find_latest_input_csv(INTERIM_DIR)
    output_csv = make_output_path(input_csv)

    df = pd.read_csv(input_csv)
    validate_input(df)

    df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    out_df = pd.DataFrame([canonicalize_row(row) for _, row in df.iterrows()])

    out_df.to_csv(output_csv, index=False)

    print(f"Input file:  {input_csv.resolve()}")
    print(f"Output file: {output_csv.resolve()}")
    print(f"Saved {len(out_df)} cleaned rows")


if __name__ == "__main__":
    main()