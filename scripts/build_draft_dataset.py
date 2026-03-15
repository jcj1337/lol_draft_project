# Will output a csv table where:
# Rows are each match
# Columns are details (champions 1-10, blue w/l)
# Samples from NA

from __future__ import annotations

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
TARGET_MATCHES = 30000    # Change depending on size we want, (upper limit)
SEED_PLAYERS = 5000     # Number of players to find (size related)
MATCH_IDS_PER_PLAYER = 30 # How many matches/player
RANDOM_SEED = 42

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLATFORM_HOST = "na1.api.riotgames.com"

REGION_HOST = "americas.api.riotgames.com"


def load_api_key() -> str:
    """ 
    Get api key from .env
    """
    load_dotenv()
    api_key = os.getenv("RIOT_API_KEY")
    if not api_key:
        raise RuntimeError("no api key found")
    return api_key


API_KEY = load_api_key()    

# set api key to default header 
session = requests.Session()
session.headers.update({"X-Riot-Token": API_KEY})


def riot_get(url: str, params: dict[str, Any] | None = None, max_retries: int = 8) -> Any:
    """
    Wrapper, Sends a GET request and handles common problems when calling 
    (rate limited/server side issues)
    """
    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=30)

        if r.status_code == 429: # if rate limited
            retry_after = int(r.headers.get("Retry-After", "2")) # in seconds
            time.sleep(retry_after + 1)
            continue
        
        if 500 <= r.status_code < 600: # if server side issue
            time.sleep(min(2 ** attempt, 30)) # wait 30s 
            continue

        r.raise_for_status()
        # raw http data to json so can use
        return r.json()
    raise RuntimeError(f"Failed after retries: {url}")


def get_current_patch(realm: str) -> str:
    """
    Returns patch in "major.minor" form, e.g. '16.5'
    """
    url = f"https://ddragon.leagueoflegends.com/realms/{realm}.json"
    data = requests.get(url, timeout=30).json()
    full_version = data["v"]  # e.g. 16.5.1
    return ".".join(full_version.split(".")[:2])

# Functions below are for player entries, need to divide emerald-diamond with master+
# cuz cringe api 

def get_division_entries(queue: str, tier: str, division: str, page: int) -> list[dict[str, Any]]:
    """ 
    Get player data for emerald-diamond players
    """
    url = f"https://{PLATFORM_HOST}/lol/league/v4/entries/{queue}/{tier}/{division}"
    return riot_get(url, params={"page": page})


def get_apex_entries(queue: str, tier: str) -> list[dict[str, Any]]:
    """
    Get player data for master-chall 
    """
    endpoint = {
        "MASTER": "masterleagues",
        "GRANDMASTER": "grandmasterleagues",
        "CHALLENGER": "challengerleagues",
    }[tier]
    url = f"https://{PLATFORM_HOST}/lol/league/v4/{endpoint}/by-queue/{queue}"
    data = riot_get(url)
    return data.get("entries", [])

def collect_seed_puuids(queue: str, target_players: int) -> list[str]:
    """
    Sample seed players from emerald+ ranked solo/duo 
    Get the initial set of player ids to use as starting points to find matches
    """
    rng = random.Random(RANDOM_SEED)
    puuids: set[str] = set()

    # emerald - diamond # ADD BACK IN EMERALD LATER
    for tier in ["DIAMOND"]:
        for division in ["I", "II", "III", "IV"]:
            page = 1
            while len(puuids) < target_players:
                entries = get_division_entries(queue, tier, division, page)
                if not entries: # if none left
                    break

                for entry in entries:
                    puuid = entry.get("puuid")
                    if puuid:
                        puuids.add(puuid)

                page += 1
                #if page > 10: # remove when move to a larger scale
                  #  break
    # masters +
    for tier in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
        for entry in get_apex_entries(queue, tier):
            puuid = entry.get("puuid")
            if puuid:
                puuids.add(puuid)

    puuids = list(puuids)
    # shuffle to mix the ranks although truthfully a bit scuffed
    rng.shuffle(puuids)
    return puuids[:target_players]

def get_ranked_match_ids(puuid: str, count: int) -> list[str]:
    """ 
    Get solo/duo match ids
    """
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
    Get match details (champions on each team and winner)
    """
    url = f"https://{REGION_HOST}/lol/match/v5/matches/{match_id}"
    return riot_get(url)


ROLE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

def normalize_team_champs(participants: list[dict[str, Any]], team_id: int) -> list[str]:
    champs_by_role: dict[str, str] = {}

    for p in participants:
        if p["teamId"] != team_id:
            continue

        role = p.get("teamPosition", "")
        champ = p["championName"]

        if role not in ROLE_ORDER:
            raise ValueError(f"Unexpected or missing role {role!r} for team {team_id}")

        if role in champs_by_role:
            raise ValueError(f"Duplicate role {role} for team {team_id}")

        champs_by_role[role] = champ

    missing = [role for role in ROLE_ORDER if role not in champs_by_role]
    if missing:
        raise ValueError(f"Missing roles for team {team_id}: {missing}")

    return [champs_by_role[role] for role in ROLE_ORDER]


def extract_row(match: dict[str, Any], patch_major_minor: str) -> dict[str, Any] | None:
    info = match["info"]

    match_patch = ".".join(info["gameVersion"].split(".")[:2])
    # removal conditionals:
    if match_patch != patch_major_minor:
        return None

    if info.get("queueId") != QUEUE_ID:
        return None

    # Remove remakes/very short games 
    if info.get("gameDuration", 0) < 600:
        return None

    participants = info["participants"]
    if len(participants) != 10:
        return None

    # 100 for blue 200 for red
    blue = normalize_team_champs(participants, 100)
    red = normalize_team_champs(participants, 200)

    blue_team = next(t for t in info["teams"] if t["teamId"] == 100)
    blue_win = int(bool(blue_team["win"]))

    return {
        "match_id": match["metadata"]["matchId"],
        "patch": match_patch,
        "blue_top": blue[0],
        "blue_jg": blue[1],
        "blue_mid": blue[2],
        "blue_adc": blue[3],
        "blue_sup": blue[4],
        "red_top": red[0],
        "red_jg": red[1],
        "red_mid": red[2],
        "red_adc": red[3],
        "red_sup": red[4],
        "blue_win": blue_win,
    }


def build_dataset() -> pd.DataFrame:
    """
    Build the dataframe 
    """
    random.seed(RANDOM_SEED)

    current_patch = get_current_patch(REALM)
    print(f"Current patch: {current_patch}")

    seed_puuids = collect_seed_puuids(
        queue=QUEUE,
        target_players=SEED_PLAYERS,
    )
    print(f"Collected {len(seed_puuids)} NA Emerald+ seed players")

    candidate_match_ids: set[str] = set()

    for i, puuid in enumerate(seed_puuids, start=1):
        try:
            ids = get_ranked_match_ids(puuid, MATCH_IDS_PER_PLAYER)
        except requests.HTTPError:
            continue

        candidate_match_ids.update(ids)

        if i % 25 == 0 or i == len(seed_puuids):
            print(f"[seed players] {i}/{len(seed_puuids)} -> {len(candidate_match_ids)} unique match ids")

        # oversample because some matches will be filtered out later
        if len(candidate_match_ids) >= int(TARGET_MATCHES * 1.3):
            break

    print(f"Fetched {len(candidate_match_ids)} candidate match ids")

    rows: list[dict[str, Any]] = []
    seen_matches: set[str] = set()

    for j, match_id in enumerate(candidate_match_ids, start=1):
        if match_id in seen_matches:
            continue
        seen_matches.add(match_id)

        try:
            match = get_match(match_id)
            row = extract_row(match, current_patch)
        except (requests.HTTPError, KeyError, ValueError):
            continue

        if row is not None:
            rows.append(row)

        if j % 100 == 0 or j == len(candidate_match_ids):
            print(f"[match details] {j}/{len(candidate_match_ids)} -> {len(rows)} usable rows")

        if len(rows) >= TARGET_MATCHES:
            break

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = build_dataset()
    out_path = OUT_DIR / f"draft_dataset_na_emeraldplus_latest_patch_{len(df)}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head())