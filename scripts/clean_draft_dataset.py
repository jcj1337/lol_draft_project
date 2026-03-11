from pathlib import Path
import pandas as pd

# paths
INTERIM_DIR = Path("data/processed")
PROCESSED_DIR = Path("data/cleaned")

BLUE_COLS = [f"blue_{i}" for i in range(1, 6)]
RED_COLS = [f"red_{i}" for i in range(1, 6)]
REQUIRED_COLS = ["match_id", "patch", "blue_win", *BLUE_COLS, *RED_COLS]

def find_latest_input_csv(data_dir: Path) -> Path:
    """
    Find the most recently modified draft dataset CSV in data/processed
    """
    candidates = list(data_dir.glob("draft_dataset*.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"No matching draft dataset CSVs found in {data_dir.resolve()}"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def make_output_path(input_csv: Path) -> Path:
    """
    Save cleaned invariant file into data/cleaned with '_cleaned' appended.
    """
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

    for cols, side in [(BLUE_COLS, "blue"), (RED_COLS, "red")]:
        if df[cols].isna().any().any():
            raise ValueError(f"Found missing champion names on {side} side.")


def canonicalize_row(row: pd.Series) -> dict:
    blue_team = tuple(sorted(str(row[col]).strip() for col in BLUE_COLS))
    red_team = tuple(sorted(str(row[col]).strip() for col in RED_COLS))
    blue_win = int(row["blue_win"])

    if len(set(blue_team)) != 5:
        raise ValueError(f"Blue team does not have 5 unique champs in match {row['match_id']}")
    if len(set(red_team)) != 5:
        raise ValueError(f"Red team does not have 5 unique champs in match {row['match_id']}")
    if set(blue_team) & set(red_team):
        raise ValueError(f"Champion overlap between teams in match {row['match_id']}")

    if blue_team <= red_team:
        team_a = blue_team
        team_b = red_team
        team_a_win = blue_win
    else:
        team_a = red_team
        team_b = blue_team
        team_a_win = 1 - blue_win

    return {
        "match_id": row["match_id"],
        "patch": row["patch"],
        "team_a_1": team_a[0],
        "team_a_2": team_a[1],
        "team_a_3": team_a[2],
        "team_a_4": team_a[3],
        "team_a_5": team_a[4],
        "team_b_1": team_b[0],
        "team_b_2": team_b[1],
        "team_b_3": team_b[2],
        "team_b_4": team_b[3],
        "team_b_5": team_b[4],
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