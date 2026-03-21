# Will output a csv table where:
# Rows are each match
# Columns are details (champions 1-10, blue w/l, player win-rate features)
# Samples from multiple LoL regions

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import json
import os
import random
import time
import pandas as pd
import requests
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Working across multiple regions 
# Riot uses:
# - platform routing for league/summoner style endpoints
# - regional routing for match-v5 endpoints
# - realm strings for current patch lookup
# - can go to the max of rate limits by going in parallel across regions, then combining results at the end for final dataset

SHARDS = [
    {"platform": "NA1",  "region": "AMERICAS", "realm": "na"},
    {"platform": "EUW1", "region": "EUROPE",   "realm": "euw"},
    {"platform": "KR",   "region": "ASIA",     "realm": "kr"},
    {"platform": "OC1",  "region": "SEA",      "realm": "oce"}
]

# Effectively we have an upper limit/max with TARGET_MATCHES
# We find SEED_PLAYERS * MATCH_IDS_PER_PLAYER matches for rows
# We later delete duplicate IDs in case that happens
# This process ensures we have unique matches

QUEUE = "RANKED_SOLO_5x5"
QUEUE_ID = 420          # Ranked solo/duo id
TARGET_MATCHES = 110000 # Total target across all shards combined
SEED_PLAYERS = 2500    # Number of players/shard to find (size related)
MATCH_IDS_PER_PLAYER = 30  # How many matches/player
RANDOM_SEED = 42
ALLOWED_PATCHES = {"16.5", "16.6"}
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MISSING_WR_VALUE = 0.5  # neutral fallback if current ranked info is missing
ROLE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


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


# one session per worker thread
_thread_local = threading.local()

def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"X-Riot-Token": API_KEY})
        _thread_local.session = s
    return _thread_local.session

def platform_host(platform: str) -> str:
    return f"{platform.lower()}.api.riotgames.com"


def region_host(region: str) -> str:
    return f"{region.lower()}.api.riotgames.com"


def cache_path_for_platform(platform: str) -> Path:
    return OUT_DIR / f"player_rank_cache_{platform.lower()}.json"


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def riot_get(url: str, params: dict[str, Any] | None = None, max_retries: int = 8) -> Any:
    """
    Wrapper, Sends a GET request and handles common problems when calling
    (rate limited/server side issues)
    """
    for attempt in range(max_retries):
        session = get_session()
        r = session.get(url, params=params, timeout=30)

        if r.status_code == 429:  # if rate limited
            retry_after = int(r.headers.get("Retry-After", "2"))  # in seconds
            print(f"Rate limited on {url} - sleeping {retry_after + 1}s")
            time.sleep(retry_after + 1)
            continue

        if 500 <= r.status_code < 600:  # if server side issue
            time.sleep(min(2 ** attempt, 30))  # wait 30s
            continue

        r.raise_for_status()
        # raw http data to json so can use
        return r.json()

    raise RuntimeError(f"Failed after retries: {url}")


def get_current_patch(realm: str) -> str:
    """
    Returns patch in "major.minor" form, e.g. '16.5'
    Patch lookup is shard-specific because current patch can differ by region
    """
    url = f"https://ddragon.leagueoflegends.com/realms/{realm}.json"
    data = requests.get(url, timeout=30).json()
    full_version = data["v"]  # e.g. 16.5.1
    return ".".join(full_version.split(".")[:2])


# Functions below are for player entries, need to divide diamond with master+
# cuz cringe api

def get_division_entries(
    queue: str,
    tier: str,
    division: str,
    page: int,
    platform: str,
) -> list[dict[str, Any]]:
    """
    Get player data for diamond players on a given platform
    """
    url = f"https://{platform_host(platform)}/lol/league/v4/entries/{queue}/{tier}/{division}"
    return riot_get(url, params={"page": page})


def get_apex_entries(queue: str, tier: str, platform: str) -> list[dict[str, Any]]:
    """
    Get player data for master/grandmaster/challenger on a given platform
    """
    endpoint = {
        "MASTER": "masterleagues",
        "GRANDMASTER": "grandmasterleagues",
        "CHALLENGER": "challengerleagues",
    }[tier]
    url = f"https://{platform_host(platform)}/lol/league/v4/{endpoint}/by-queue/{queue}"
    data = riot_get(url)
    return data.get("entries", [])


def load_player_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: cache file {cache_path} is invalid. Starting with empty cache.")
            return {}
    return {}


def save_player_cache(cache: dict[str, dict[str, Any]], cache_path: Path) -> None:
    temp_path = cache_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    temp_path.replace(cache_path)


def collect_seed_puuids(
    queue: str,
    target_players: int,
    cache: dict[str, dict[str, Any]],
    platform: str,
) -> list[str]:
    """
    Sample seed players from Diamond+ ranked solo/duo
    Get the initial set of player ids to use as starting points to find matches

    Also prefill the cache using the league entry data we already get here,
    so we avoid some later by-puuid lookups for skill features
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

    # diamond
    for tier in ["DIAMOND"]:
        for division in ["I", "II", "III", "IV"]:
            page = 1
            while len(puuids) < target_players:
                entries = get_division_entries(queue, tier, division, page, platform)
                if not entries:  # if none left
                    break

                for entry in entries:
                    puuid = entry.get("puuid")
                    if puuid:
                        puuids.add(puuid)
                        maybe_cache_entry(entry)

                page += 1

    # masters +
    for tier in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
        for entry in get_apex_entries(queue, tier, platform):
            puuid = entry.get("puuid")
            if puuid:
                puuids.add(puuid)
                maybe_cache_entry(entry)

    puuids = list(puuids)
    # shuffle to mix the ranks although truthfully a bit scuffed
    rng.shuffle(puuids)
    return puuids[:target_players]


def get_ranked_match_ids(puuid: str, count: int, region: str) -> list[str]:
    """
    Get solo/duo match ids
    Small sleep to avoid bursting too hard into regional rate limits
    """
    time.sleep(1)
    url = f"https://{region_host(region)}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {
        "start": 0,
        "count": count,
        "queue": QUEUE_ID,
        "type": "ranked",
    }
    return riot_get(url, params=params)


def get_match(match_id: str, region: str) -> dict[str, Any]:
    """
    Get match details (champions on each team and winner)
    """
    url = f"https://{region_host(region)}/lol/match/v5/matches/{match_id}"
    return riot_get(url)


def get_ranked_entries_by_puuid(puuid: str, platform: str) -> list[dict[str, Any]]:
    """
    Get current ranked entries for a player directly by PUUID
    Small sleep to avoid bursting too hard into platform rate limits
    """
    time.sleep(1.21)
    url = f"https://{platform_host(platform)}/lol/league/v4/entries/by-puuid/{puuid}"
    return riot_get(url)


def get_player_ranked_profile(
    puuid: str,
    cache: dict[str, dict[str, Any]],
    platform: str,
) -> dict[str, Any]:
    """
    Returns current ranked solo queue stats for a player, cached by PUUID.
    """
    if puuid in cache:
        return cache[puuid]

    try:
        entries = get_ranked_entries_by_puuid(puuid, platform)
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
    """
    Return the 5 participants on a team in fixed role order
    """
    participants_by_role: dict[str, dict[str, Any]] = {}

    for p in participants:
        if p["teamId"] != team_id:
            continue

        # Use teamPosition first; fall back to individualPosition if needed
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
    platform: str,
    region: str,
) -> dict[str, Any] | None:
    info = match["info"]

    match_patch = ".".join(info["gameVersion"].split(".")[:2])

    # removal conditionals:
    #if match_patch not in ALLOWED_PATCHES:
      #  return None

    if info.get("queueId") != QUEUE_ID:
        return None

    # Remove remakes/very short games
    if info.get("gameDuration", 0) < 600:
        return None

    participants = info["participants"]
    if len(participants) != 10:
        return None

    # 100 for blue 200 for red
    blue_participants = normalize_team_participants(participants, 100)
    red_participants = normalize_team_participants(participants, 200)

    blue_team = next(t for t in info["teams"] if t["teamId"] == 100)
    blue_win = int(bool(blue_team["win"]))

    # current player-skill proxy features
    blue_profiles = [get_player_ranked_profile(p["puuid"], player_cache, platform) for p in blue_participants]
    red_profiles = [get_player_ranked_profile(p["puuid"], player_cache, platform) for p in red_participants]

    blue_avg_wr = sum(p["winrate"] for p in blue_profiles) / 5
    red_avg_wr = sum(p["winrate"] for p in red_profiles) / 5

    return {
        "match_id": match["metadata"]["matchId"],
        "source_platform": platform,
        "source_region": region,
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

# Run one worker per regional routing bucket
# Match-v5 calls share the regional bucket, so this is the fastest sane parallelism
MAX_REGION_WORKERS = 4

def group_shards_by_region(shards: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for shard in shards:
        groups.setdefault(shard["region"], []).append(shard)
    return groups


def build_region_group(region_name: str, shards: list[dict[str, str]], total_target_for_group: int) -> pd.DataFrame:
    """
    Build all shards that belong to one regional routing bucket.
    """
    #  We keep shards inside a region group sequential, bc their match-v5 calls
    #  still share the same regional rate limit bucket.
    print(f"\nStarting region group {region_name} with {len(shards)} shards")

    per_shard_target = max(1, (total_target_for_group + len(shards) - 1) // len(shards))
    dfs: list[pd.DataFrame] = []

    for shard in shards:
        df_shard = build_dataset_for_shard(shard, per_shard_target)
        print(f"[{shard['platform']}] final rows: {len(df_shard)}")

        if not df_shard.empty:
            dfs.append(df_shard)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    if len(df) > total_target_for_group:
        df = df.iloc[:total_target_for_group].reset_index(drop=True)

    return df

def build_dataset_for_shard(shard: dict[str, str], target_matches: int) -> pd.DataFrame:
    """
    Build one shard-specific dataframe, then later concatenate all shards together
    """
    random.seed(RANDOM_SEED)

    platform = shard["platform"]
    region = shard["region"]
    realm = shard["realm"]

    start_time = time.time()

    current_patch = get_current_patch(realm)
    print(f"\n=== Building shard {platform}/{region} | using patch: {current_patch} ===")

    cache_path = cache_path_for_platform(platform)
    player_cache = load_player_cache(cache_path)
    print(f"Loaded {len(player_cache)} cached player ranked profiles for {platform}")

    seed_puuids = collect_seed_puuids(
        queue=QUEUE,
        target_players=SEED_PLAYERS,
        cache=player_cache,
        platform=platform,
    )
    print(f"Collected {len(seed_puuids)} {platform} Diamond+ seed players")

    candidate_match_ids: list[str] = []
    seen_candidate_ids: set[str] = set()

    for i, puuid in enumerate(seed_puuids, start=1):
        try:
            ids = get_ranked_match_ids(puuid, MATCH_IDS_PER_PLAYER, region)
        except requests.HTTPError:
            continue

        for mid in ids:
            if mid not in seen_candidate_ids:
                seen_candidate_ids.add(mid)
                candidate_match_ids.append(mid)

        if i % 25 == 0 or i == len(seed_puuids):
            print(f"[{platform} seed players] {i}/{len(seed_puuids)} -> {len(candidate_match_ids)} unique match ids")

        # oversample because some matches will be filtered out later
        if len(candidate_match_ids) >= int(target_matches * 1.3):
            break

    print(f"[{platform}] Fetched {len(candidate_match_ids)} candidate match ids")

    rows: list[dict[str, Any]] = []
    seen_matches: set[str] = set()
    skip_count = 0

    for j, match_id in enumerate(candidate_match_ids, start=1):
        if match_id in seen_matches:
            continue
        seen_matches.add(match_id)

        try:
            match = get_match(match_id, region)
            row = extract_row(match, current_patch, player_cache, platform, region)
        except (requests.HTTPError, KeyError, ValueError):
            skip_count += 1
            continue

        if row is not None:
            rows.append(row)
        else:
            skip_count += 1

        if j % 100 == 0 or len(rows) >= target_matches or j == len(candidate_match_ids):
            elapsed = time.time() - start_time
            matches_per_sec = j / elapsed if elapsed > 0 else 0.0
            rows_per_sec = len(rows) / elapsed if elapsed > 0 else 0.0

            remaining_matches = len(candidate_match_ids) - j
            eta_by_match_scan = remaining_matches / matches_per_sec if matches_per_sec > 0 else float("inf")

            remaining_rows = max(target_matches - len(rows), 0)
            eta_by_row_rate = remaining_rows / rows_per_sec if rows_per_sec > 0 else float("inf")

            print(
                f"[{platform} match details] {j}/{len(candidate_match_ids)} -> {len(rows)} usable rows | "
                f"skipped={skip_count} | "
                f"elapsed={format_seconds(elapsed)} | "
                f"scan_rate={matches_per_sec:.2f} matches/s | "
                f"row_rate={rows_per_sec:.2f} rows/s | "
                f"eta_scan={format_seconds(eta_by_match_scan) if eta_by_match_scan != float('inf') else 'N/A'} | "
                f"eta_rows={format_seconds(eta_by_row_rate) if eta_by_row_rate != float('inf') else 'N/A'}"
            )

        if j % 200 == 0:
            save_player_cache(player_cache, cache_path)
            print(f"Saved player cache for {platform} with {len(player_cache)} entries")

        if len(rows) >= target_matches:
            break

    save_player_cache(player_cache, cache_path)

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    return df


def build_dataset() -> pd.DataFrame:
    """
    Use all shards to build the full multiregion dataframe in parallel by regional routing group.
    """
    region_groups = group_shards_by_region(SHARDS)

    # Split the total target roughly evenly across the four regional routing groups
    per_region_target = max(1, (TARGET_MATCHES + len(region_groups) - 1) // len(region_groups))

    dfs: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=min(MAX_REGION_WORKERS, len(region_groups))) as executor:
        future_to_region = {
            executor.submit(build_region_group, region_name, shards, per_region_target): region_name
            for region_name, shards in region_groups.items()
        }

        for future in as_completed(future_to_region):
            region_name = future_to_region[future]

            try:
                df_region = future.result()
            except Exception as e:
                print(f"[{region_name}] failed: {type(e).__name__}: {e}")
                continue

            print(f"[{region_name}] total rows collected: {len(df_region)}")

            if not df_region.empty:
                dfs.append(df_region)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    # trim back down to exact requested size if we overshoot
    if len(df) > TARGET_MATCHES:
        df = df.iloc[:TARGET_MATCHES].reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = build_dataset()
    out_path = OUT_DIR / f"draft_dataset_multiregion_diamondplus_{len(df)}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head())