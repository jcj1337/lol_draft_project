# Will output a csv table where:
# Rows are each match
# Columns are details (champions 1-10, blue w/l)
# Samples from NA

from __future__ import annotations
import json
import os
import random
import time
import pandas as pd
import requests
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Working in NA but maybe change down the line 

PLATFORM = "NA1"        # NA server for league/summoner endpoints
REGION = "AMERICAS"     # Routing cluster for NA match-v5 endpoints
REALM = "na"            # For latest patch lookup later 

# Effectively we have an upper limit/max with TARGET_MATCHES
# We find SEED_PLAYERS * MATCH_IDS_PER_PLAYER matches for rows
# We later delete duplicate IDs in case that happens
# This process ensures we have unique matches 

QUEUE = "RANKED_SOLO_5x5"
QUEUE_ID = 420          # Ranked solo/duo id 
TARGET_MATCHES = 15000  # Change depending on size we want, (upper limit)
SEED_PLAYERS = 2000    # Number of players to find (size related)
MATCH_IDS_PER_PLAYER = 20 # How many matches/player
RANDOM_SEED = 42

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLATFORM_HOST = "na1.api.riotgames.com"
REGION_HOST = "americas.api.riotgames.com"

CACHE_PATH = OUT_DIR / "player_rank_cache.json"
MISSING_WR_VALUE = 0.5

ROLE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("RIOT_API_KEY")
    if not api_key:
        raise RuntimeError("no api key found")
    return api_key


API_KEY = load_api_key()

session = requests.Session()
session.headers.update({"X-Riot-Token": API_KEY})


def riot_get(url: str, params: dict[str, Any] | None = None, max_retries: int = 8) -> Any:
    """
    Wrapper, sends a GET request and handles common problems when calling
    (rate limited/server side issues)
    """
    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=30)

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "2"))
            print(f"Rate limited on {url} - sleeping {retry_after + 1}s")
            time.sleep(retry_after + 1)
            continue

        if 500 <= r.status_code < 600:
            time.sleep(min(2 ** attempt, 30))
            continue

        r.raise_for_status()
        return r.json()

    raise RuntimeError(f"Failed after retries: {url}")


def get_current_patch(realm: str) -> str:
    """
    Returns patch in 'major.minor' form, e.g. '16.5'
    """
    url = f"https://ddragon.leagueoflegends.com/realms/{realm}.json"
    data = requests.get(url, timeout=30).json()
    full_version = data["v"]
    return ".".join(full_version.split(".")[:2])


def get_division_entries(queue: str, tier: str, division: str, page: int) -> list[dict[str, Any]]:
    """
    Get player data for emerald-diamond players
    """
    url = f"https://{PLATFORM_HOST}/lol/league/v4/entries/{queue}/{tier}/{division}"
    return riot_get(url, params={"page": page})


def get_apex_entries(queue: str, tier: str) -> list[dict[str, Any]]:
    """
    Get player data for master/grandmaster/challenger
    """
    endpoint = {
        "MASTER": "masterleagues",
        "GRANDMASTER": "grandmasterleagues",
        "CHALLENGER": "challengerleagues",
    }[tier]
    url = f"https://{PLATFORM_HOST}/lol/league/v4/{endpoint}/by-queue/{queue}"
    data = riot_get(url)
    return data.get("entries", [])


def collect_seed_puuids(queue: str, target_players: int, cache: dict[str, dict[str, Any]]) -> list[str]:
    """
    Sample seed players from Diamond+ ranked solo/duo and prefill ranked-profile cache.
    """
    rng = random.Random(RANDOM_SEED)
    puuids: set[str] = set()

    def maybe_cache_entry(entry: dict[str, Any]) -> None:
        puuid = entry.get("puuid")
        if not puuid:
            return

        wins = int(entry.get("wins", 0))
        losses = int(entry.get("losses", 0))
        games = wins + losses
        winrate = wins / games if games > 0 else MISSING_WR_VALUE

        cache[puuid] = {
            "wins": wins,
            "losses": losses,
            "games": games,
            "winrate": winrate,
            "tier": entry.get("tier"),
            "rank": entry.get("rank"),
            "lp": entry.get("leaguePoints"),
        }

    for tier in ["DIAMOND"]:
        for division in ["I", "II", "III", "IV"]:
            page = 1
            while len(puuids) < target_players:
                entries = get_division_entries(queue, tier, division, page)
                if not entries:
                    break

                for entry in entries:
                    puuid = entry.get("puuid")
                    if puuid:
                        puuids.add(puuid)
                        maybe_cache_entry(entry)

                page += 1

    for tier in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
        for entry in get_apex_entries(queue, tier):
            puuid = entry.get("puuid")
            if puuid:
                puuids.add(puuid)
                maybe_cache_entry(entry)

    puuids = list(puuids)
    rng.shuffle(puuids)
    return puuids[:target_players]


def get_ranked_match_ids(puuid: str, count: int) -> list[str]:
    """
    Get solo/duo match ids
    """
    time.sleep(1.25)
    url = f"https://{REGION_HOST}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {
        "start": 0,
        "count": count,
        "queue": QUEUE_ID,
        "type": "ranked",
    }
    return riot_get(url, params=params)


def get_match(match_id: str) -> dict[str, Any]:
    """
    Get match details
    """
    url = f"https://{REGION_HOST}/lol/match/v5/matches/{match_id}"
    return riot_get(url)


def get_ranked_entries_by_puuid(puuid: str) -> list[dict[str, Any]]:
    """
    Get all ranked entries for a player directly by PUUID.
    """
    time.sleep(1.25)  # try and prevent hitting rate limits perma
    url = f"https://{PLATFORM_HOST}/lol/league/v4/entries/by-puuid/{puuid}"
    return riot_get(url)


def load_player_cache() -> dict[str, dict[str, Any]]:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: cache file {CACHE_PATH} is invalid. Starting with empty cache.")
            return {}
    return {}


def save_player_cache(cache: dict[str, dict[str, Any]]) -> None:
    temp_path = CACHE_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    temp_path.replace(CACHE_PATH)

def get_player_ranked_profile(puuid: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Returns current ranked solo queue stats for a player, cached by PUUID.
    """
    if puuid in cache:
        return cache[puuid]

    try:
        entries = get_ranked_entries_by_puuid(puuid)
    except requests.HTTPError:
        profile = {
            "wins": 0,
            "losses": 0,
            "games": 0,
            "winrate": MISSING_WR_VALUE,
            "tier": None,
            "rank": None,
            "lp": None,
        }
        cache[puuid] = profile
        return profile

    solo_entry = next(
        (e for e in entries if e.get("queueType") == "RANKED_SOLO_5x5"),
        None,
    )

    if solo_entry is None:
        profile = {
            "wins": 0,
            "losses": 0,
            "games": 0,
            "winrate": MISSING_WR_VALUE,
            "tier": None,
            "rank": None,
            "lp": None,
        }
    else:
        wins = int(solo_entry.get("wins", 0))
        losses = int(solo_entry.get("losses", 0))
        games = wins + losses
        winrate = wins / games if games > 0 else MISSING_WR_VALUE

        profile = {
            "wins": wins,
            "losses": losses,
            "games": games,
            "winrate": winrate,
            "tier": solo_entry.get("tier"),
            "rank": solo_entry.get("rank"),
            "lp": solo_entry.get("leaguePoints"),
        }

    cache[puuid] = profile
    return profile


def normalize_team_participants(participants: list[dict[str, Any]], team_id: int) -> list[dict[str, Any]]:
    participants_by_role: dict[str, dict[str, Any]] = {}

    for p in participants:
        if p["teamId"] != team_id:
            continue

        role = p.get("teamPosition", "")
        if role not in ROLE_ORDER:
            role = p.get("individualPosition", "")

        if role not in ROLE_ORDER:
            raise ValueError(f"Unexpected or missing role {role!r} for team {team_id}")

        if role in participants_by_role:
            raise ValueError(f"Duplicate role {role} for team {team_id}")

        participants_by_role[role] = p

    missing = [role for role in ROLE_ORDER if role not in participants_by_role]
    if missing:
        raise ValueError(f"Missing roles for team {team_id}: {missing}")

    return [participants_by_role[role] for role in ROLE_ORDER]


def extract_row(
    match: dict[str, Any],
    patch_major_minor: str,
    player_cache: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    info = match["info"]

    match_patch = ".".join(info["gameVersion"].split(".")[:2])

   # if match_patch != patch_major_minor:
     #   return None

    if info.get("queueId") != QUEUE_ID:
        return None

    if info.get("gameDuration", 0) < 600:
        return None

    participants = info["participants"]
    if len(participants) != 10:
        return None

    blue_participants = normalize_team_participants(participants, 100)
    red_participants = normalize_team_participants(participants, 200)

    blue_team = next(t for t in info["teams"] if t["teamId"] == 100)
    blue_win = int(bool(blue_team["win"]))

    blue_profiles = [get_player_ranked_profile(p["puuid"], player_cache) for p in blue_participants]
    red_profiles = [get_player_ranked_profile(p["puuid"], player_cache) for p in red_participants]

    blue_avg_wr = sum(p["winrate"] for p in blue_profiles) / 5
    red_avg_wr = sum(p["winrate"] for p in red_profiles) / 5

    return {
        "match_id": match["metadata"]["matchId"],
        "patch": match_patch,

        "blue_top": blue_participants[0]["championName"],
        "blue_jg": blue_participants[1]["championName"],
        "blue_mid": blue_participants[2]["championName"],
        "blue_adc": blue_participants[3]["championName"],
        "blue_sup": blue_participants[4]["championName"],

        "red_top": red_participants[0]["championName"],
        "red_jg": red_participants[1]["championName"],
        "red_mid": red_participants[2]["championName"],
        "red_adc": red_participants[3]["championName"],
        "red_sup": red_participants[4]["championName"],

        "blue_top_wr": blue_profiles[0]["winrate"],
        "blue_jg_wr": blue_profiles[1]["winrate"],
        "blue_mid_wr": blue_profiles[2]["winrate"],
        "blue_adc_wr": blue_profiles[3]["winrate"],
        "blue_sup_wr": blue_profiles[4]["winrate"],

        "red_top_wr": red_profiles[0]["winrate"],
        "red_jg_wr": red_profiles[1]["winrate"],
        "red_mid_wr": red_profiles[2]["winrate"],
        "red_adc_wr": red_profiles[3]["winrate"],
        "red_sup_wr": red_profiles[4]["winrate"],

        "blue_top_games": blue_profiles[0]["games"],
        "blue_jg_games": blue_profiles[1]["games"],
        "blue_mid_games": blue_profiles[2]["games"],
        "blue_adc_games": blue_profiles[3]["games"],
        "blue_sup_games": blue_profiles[4]["games"],

        "red_top_games": red_profiles[0]["games"],
        "red_jg_games": red_profiles[1]["games"],
        "red_mid_games": red_profiles[2]["games"],
        "red_adc_games": red_profiles[3]["games"],
        "red_sup_games": red_profiles[4]["games"],

        "blue_avg_wr": blue_avg_wr,
        "red_avg_wr": red_avg_wr,
        "avg_wr_diff": blue_avg_wr - red_avg_wr,

        "blue_win": blue_win,
    }

def format_seconds(seconds: float) -> str:
    """
    time format for logging 
    """
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def build_dataset() -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    start_time = time.time()
    current_patch = get_current_patch(REALM)
    print(f"Current patch: {current_patch}")

    player_cache = load_player_cache()
    print(f"Loaded {len(player_cache)} cached player ranked profiles")

    seed_puuids = collect_seed_puuids(
        queue=QUEUE,
        target_players=SEED_PLAYERS,
        cache=player_cache,
    )
    print(f"Collected {len(seed_puuids)} NA Diamond+ seed players")

    candidate_match_ids: list[str] = []
    seen_candidate_ids: set[str] = set()

    for i, puuid in enumerate(seed_puuids, start=1):
        try:
            ids = get_ranked_match_ids(puuid, MATCH_IDS_PER_PLAYER)
        except (requests.HTTPError, KeyError, ValueError) as e:
            continue

        for mid in ids:
            if mid not in seen_candidate_ids:
                seen_candidate_ids.add(mid)
                candidate_match_ids.append(mid)

        if i % 25 == 0 or i == len(seed_puuids):
            print(f"[seed players] {i}/{len(seed_puuids)} -> {len(candidate_match_ids)} unique match ids")

        if len(candidate_match_ids) >= int(TARGET_MATCHES * 1.3):
            break

    print(f"Fetched {len(candidate_match_ids)} candidate match ids")

    rows: list[dict[str, Any]] = []
    seen_matches: set[str] = set()

    for j, match_id in enumerate(candidate_match_ids, start=1):
        if match_id in seen_matches:
            continue
        seen_matches.add(match_id)

        if j % 50 == 0 or j <= 5:
            print(f"Processing match {j}/{len(candidate_match_ids)}: {match_id}")
        try:
            match = get_match(match_id)
            row = extract_row(match, current_patch, player_cache)
        except (requests.HTTPError, KeyError, ValueError) as e:
            print(f"Skipping {match_id}: {type(e).__name__}: {e}")
            continue

        if row is not None:
            rows.append(row)

        if j % 100 == 0 or j == len(candidate_match_ids):
            elapsed = time.time() - start_time
            matches_per_sec = j / elapsed if elapsed > 0 else 0.0
            rows_per_sec = len(rows) / elapsed if elapsed > 0 else 0.0

            remaining_matches = len(candidate_match_ids) - j
            eta_by_match_scan = remaining_matches / matches_per_sec if matches_per_sec > 0 else float("inf")

            remaining_rows = max(TARGET_MATCHES - len(rows), 0)
            eta_by_row_rate = remaining_rows / rows_per_sec if rows_per_sec > 0 else float("inf")

            print(
                f"[match details] {j}/{len(candidate_match_ids)} -> {len(rows)} usable rows | "
                f"elapsed={format_seconds(elapsed)} | "
                f"scan_rate={matches_per_sec:.2f} matches/s | "
                f"row_rate={rows_per_sec:.2f} rows/s | "
                f"eta_scan={format_seconds(eta_by_match_scan) if eta_by_match_scan != float('inf') else 'N/A'} | "
                f"eta_rows={format_seconds(eta_by_row_rate) if eta_by_row_rate != float('inf') else 'N/A'}"
            )

        if j % 200 == 0:
            save_player_cache(player_cache)
            print(f"Saved player cache with {len(player_cache)} entries")

        if len(rows) >= TARGET_MATCHES:
            break

    save_player_cache(player_cache)

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = build_dataset()
    out_path = OUT_DIR / f"draft_dataset_na_diamondplus_latest_patch_{len(df)}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head())