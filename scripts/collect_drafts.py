"""
Draft-only match collector for ranked solo/duo, crawling Diamond+ players (default) or
Emerald players (scripts.collect_emerald, run on a second machine with its own key).

How it works:
  1. Roster threads (one per platform, e.g. NA1) download the full Diamond+ ladder once a
     day, plus Emerald I-IV when crawling Emerald. League-v4 runs on the platform rate
     limit, separate from match-v5.
  2. Region threads (one per match-v5 routing region, e.g. AMERICAS) crawl roster players
     of the crawled ladder: list their ranked match ids since `--since` (or since their last
     crawl), fetch matches we have not seen, and store one row per match in SQLite as soon
     as it arrives.
  3. When the API key expires, everything pauses until a new key appears in .env.

Everything is resumable: stop with Ctrl+C at any time and start again later. Each ladder
has its own database, so the Emerald collector never touches the Diamond+ one.

Run from the project root:
    python -m scripts.collect_drafts collect [--since YYYY-MM-DD]
    python -m scripts.collect_drafts status
    python -m scripts.collect_drafts export [--out path.csv]   # every game; optional --min-diamond / --min-emerald-plus
    python -m scripts.collect_drafts merge path/to/pack.sqlite   # add games from the Emerald collector
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.riot_client import ApiKeyManager, CollectionStopped, RiotClient

log = logging.getLogger("collector")

# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("data/collector")
DB_PATH = DATA_DIR / "drafts.sqlite"
LOG_PATH = DATA_DIR / "collector.log"
EXPORT_DIR = DATA_DIR / "exports"
# games shorter than this ended early (a leaver, early surrender) and say little about the draft;
# they stay in the database but are left out of exports
MIN_EXPORT_DURATION_SECONDS = 15 * 60
ENV_PATH = Path(".env")

QUEUE = "RANKED_SOLO_5x5"
QUEUE_ID = 420

# match-v5 routing region -> platforms. Platforms in one region share its match-v5 rate
# limit, so extra platforms add supply (more Diamond+ games), not speed.
REGIONS: dict[str, list[str]] = {
    "AMERICAS": ["NA1", "BR1", "LA1", "LA2"],
    "EUROPE": ["EUW1", "EUN1", "TR1"],
    "ASIA": ["KR", "JP1"],
    "SEA": ["OC1", "SG2", "TW2", "VN2"],
}
DIVISIONS = ["I", "II", "III", "IV"]
APEX_TIERS = {
    "MASTER": "masterleagues",
    "GRANDMASTER": "grandmasterleagues",
    "CHALLENGER": "challengerleagues",
}

# ladder -> tiers whose players are crawled. The Diamond+ ladder is always downloaded, so
# n_diamond_plus means the same in every database.
LADDERS: dict[str, list[str]] = {
    "diamond_plus": ["DIAMOND", *APEX_TIERS],
    "emerald": ["EMERALD"],
}
LADDER_LABELS = {"diamond_plus": "Diamond+", "emerald": "Emerald"}
LADDER = "diamond_plus"  # switched by use_ladder()

DEFAULT_LOOKBACK_DAYS = 14
ROSTER_MAX_AGE_HOURS = 24
ROSTER_CHECK_SECONDS = 600
ROSTER_CURRENT_HOURS = 36          # players seen in a refresh this recent count as Diamond+
MIN_RECRAWL_HOURS = 6              # don't re-list an active player's matches more often than this
INACTIVE_RECRAWL_HOURS = 48        # players whose last crawl found no games are re-checked less often
RECRAWL_OVERLAP_SECONDS = 2 * 3600 # re-list a bit before the last crawl to catch games in progress
PLAYER_BATCH = 200
IDLE_WAIT_SECONDS = 60
STATUS_EVERY_SECONDS = 300

ROLE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROLES = ["top", "jg", "mid", "adc", "sup"]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    platform        TEXT NOT NULL,
    region          TEXT NOT NULL,
    game_creation   INTEGER NOT NULL,   -- unix seconds
    game_version    TEXT NOT NULL,      -- e.g. 16.6.712.1234
    patch           TEXT NOT NULL,      -- e.g. 16.6
    game_duration   INTEGER NOT NULL,   -- seconds
    {", ".join(f"blue_{r} TEXT NOT NULL" for r in ROLES)},
    {", ".join(f"red_{r} TEXT NOT NULL" for r in ROLES)},
    {", ".join(f"blue_{r}_id INTEGER NOT NULL" for r in ROLES)},
    {", ".join(f"red_{r}_id INTEGER NOT NULL" for r in ROLES)},
    blue_win        INTEGER NOT NULL,
    blue_bans       TEXT NOT NULL,      -- JSON champion ids in pick-turn order
    red_bans        TEXT NOT NULL,
    puuids          TEXT NOT NULL,      -- JSON, blue top..sup then red top..sup
    n_diamond_plus  INTEGER NOT NULL,   -- how many of the 10 players were on our Diamond+ roster
    collected_at    INTEGER NOT NULL,
    n_emerald       INTEGER,            -- how many were Emerald; NULL unless the Emerald ladder was downloaded
    crawled_from    TEXT NOT NULL DEFAULT 'diamond_plus'  -- ladder whose players led us to this game
);
CREATE INDEX IF NOT EXISTS idx_matches_patch ON matches (patch);

CREATE TABLE IF NOT EXISTS skipped (
    match_id TEXT PRIMARY KEY,
    reason   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS roster (
    platform TEXT NOT NULL,
    puuid    TEXT NOT NULL,
    tier     TEXT,
    division TEXT,
    lp       INTEGER,
    games    INTEGER,                  -- season games; only used to crawl active players first
    seen_at  INTEGER NOT NULL,
    PRIMARY KEY (platform, puuid)
);
CREATE INDEX IF NOT EXISTS idx_roster_seen ON roster (platform, seen_at);

CREATE TABLE IF NOT EXISTS progress (
    platform      TEXT NOT NULL,
    puuid         TEXT NOT NULL,
    crawled_until INTEGER,             -- unix seconds; matches before this were listed
    last_attempt  INTEGER NOT NULL,
    found         INTEGER,             -- match ids listed by the last crawl
    PRIMARY KEY (platform, puuid)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# -----------------------------
# Database
# -----------------------------
def use_ladder(ladder: str) -> None:
    """Crawl `ladder` instead of Diamond+, with its own database and log."""
    global LADDER, DB_PATH, LOG_PATH
    LADDER = ladder
    suffix = "" if ladder == "diamond_plus" else f"_{ladder}"
    DB_PATH = DATA_DIR / f"drafts{suffix}.sqlite"
    LOG_PATH = DATA_DIR / f"collector{suffix}.log"


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # columns added after the first version of the schema
    for table, column in (
        ("roster", "games INTEGER"),
        ("progress", "found INTEGER"),
        ("matches", "n_emerald INTEGER"),
        ("matches", "crawled_from TEXT NOT NULL DEFAULT 'diamond_plus'"),
    ):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column.split()[0] not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


# -----------------------------
# Match parsing
# -----------------------------
class SkipMatch(Exception):
    pass


def team_by_role(participants: list[dict[str, Any]], team_id: int) -> list[dict[str, Any]]:
    by_role: dict[str, dict[str, Any]] = {}
    for p in participants:
        if p["teamId"] != team_id:
            continue
        role = p.get("teamPosition", "")
        if role not in ROLE_ORDER:
            role = p.get("individualPosition", "")
        if role not in ROLE_ORDER or role in by_role:
            raise SkipMatch("roles")
        by_role[role] = p
    if len(by_role) != 5:
        raise SkipMatch("roles")
    return [by_role[r] for r in ROLE_ORDER]


def parse_match(match: dict[str, Any], region: str) -> dict[str, Any]:
    """One row for the matches table (without n_diamond_plus). Raises SkipMatch."""
    info = match["info"]
    if info.get("queueId") != QUEUE_ID:
        raise SkipMatch("queue")

    participants = info["participants"]
    if len(participants) != 10:
        raise SkipMatch("participants")
    if any(p.get("gameEndedInEarlySurrender") for p in participants):
        raise SkipMatch("remake")

    blue = team_by_role(participants, 100)
    red = team_by_role(participants, 200)
    teams = {t["teamId"]: t for t in info["teams"]}

    def bans(team_id: int) -> str:
        team_bans = sorted(teams[team_id].get("bans", []), key=lambda b: b.get("pickTurn", 0))
        return json.dumps([b["championId"] for b in team_bans])

    row: dict[str, Any] = {
        "match_id": match["metadata"]["matchId"],
        "platform": info["platformId"],
        "region": region,
        "game_creation": int(info["gameCreation"]) // 1000,
        "game_version": info["gameVersion"],
        "patch": ".".join(info["gameVersion"].split(".")[:2]),
        "game_duration": int(info["gameDuration"]),
        "blue_win": int(bool(teams[100]["win"])),
        "blue_bans": bans(100),
        "red_bans": bans(200),
        "puuids": json.dumps([p["puuid"] for p in blue + red]),
        "collected_at": int(time.time()),
    }
    for side, team in (("blue", blue), ("red", red)):
        for r, p in zip(ROLES, team):
            row[f"{side}_{r}"] = p["championName"]
            row[f"{side}_{r}_id"] = int(p["championId"])
    return row


# -----------------------------
# Roster (Diamond+ ladder per platform)
# -----------------------------
def refresh_roster(client: RiotClient, conn: sqlite3.Connection, platform: str) -> int:
    started = int(time.time())
    n = 0

    def store(entries: list[dict[str, Any]], tier: str) -> None:
        nonlocal n
        rows = [
            (platform, e["puuid"], tier, e.get("rank"), e.get("leaguePoints"),
             int(e.get("wins", 0)) + int(e.get("losses", 0)), started)
            for e in entries if e.get("puuid")
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO roster (platform, puuid, tier, division, lp, games, seen_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        n += len(rows)

    for tier, endpoint in APEX_TIERS.items():
        league = client.get(platform, f"/lol/league/v4/{endpoint}/by-queue/{QUEUE}", method=f"league-{endpoint}")
        store(league.get("entries", []), tier)

    for tier in ["DIAMOND"] + (["EMERALD"] if LADDER == "emerald" else []):
        for division in DIVISIONS:
            page = 1
            while True:
                entries = client.get(platform, f"/lol/league/v4/entries/{QUEUE}/{tier}/{division}", {"page": page}, method="league-entries")
                if not entries:
                    break
                store(entries, tier)
                page += 1

    set_meta(conn, f"roster_refreshed:{platform}", str(started))
    return n


def roster_worker(client: RiotClient, platform: str, stop: threading.Event) -> None:
    conn = connect()
    while not stop.is_set():
        try:
            last = int(get_meta(conn, f"roster_refreshed:{platform}") or 0)
            if time.time() - last >= ROSTER_MAX_AGE_HOURS * 3600:
                log.info("[%s] refreshing roster", platform)
                n = refresh_roster(client, conn, platform)
                log.info("[%s] roster refreshed: %d players", platform, n)
        except CollectionStopped:
            break
        except requests.HTTPError as e:
            log.error("[%s] roster refresh failed (%s); platform disabled for this run", platform, e)
            break
        except Exception:
            log.error("[%s] roster worker error:\n%s", platform, traceback.format_exc())
        stop.wait(ROSTER_CHECK_SECONDS)
    conn.close()


# -----------------------------
# Crawling (match-v5 per region)
# -----------------------------
class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stored: dict[str, int] = {}
        self.skipped: dict[str, int] = {}
        self.players: dict[str, int] = {}

    def add(self, counter: dict[str, int], region: str, n: int = 1) -> None:
        with self.lock:
            counter[region] = counter.get(region, 0) + n

    def snapshot(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        with self.lock:
            return dict(self.stored), dict(self.skipped), dict(self.players)


def next_players(conn: sqlite3.Connection, platforms: list[str]) -> list[tuple[str, str, int | None]]:
    now = int(time.time())
    placeholders = ",".join("?" * len(platforms))
    tiers = LADDERS[LADDER]
    return conn.execute(
        f"""
        SELECT r.platform, r.puuid, p.crawled_until
        FROM roster r
        LEFT JOIN progress p ON p.platform = r.platform AND p.puuid = r.puuid
        WHERE r.platform IN ({placeholders})
          AND r.tier IN ({",".join("?" * len(tiers))})
          AND r.seen_at >= ?
          AND (p.last_attempt IS NULL
               OR p.last_attempt < CASE WHEN COALESCE(p.found, 1) > 0 THEN ? ELSE ? END)
        -- never-crawled players first, most season games first; then the longest-waiting
        ORDER BY p.last_attempt IS NOT NULL, COALESCE(r.games, 0) DESC, p.last_attempt
        LIMIT ?
        """,
        (*platforms, *tiers, now - ROSTER_CURRENT_HOURS * 3600,
         now - MIN_RECRAWL_HOURS * 3600, now - INACTIVE_RECRAWL_HOURS * 3600, PLAYER_BATCH),
    ).fetchall()


def list_match_ids(client: RiotClient, region: str, puuid: str, start_time: int) -> list[str]:
    ids: list[str] = []
    start = 0
    while True:
        page = client.get(
            region,
            f"/lol/match/v5/matches/by-puuid/{puuid}/ids",
            {"startTime": start_time, "queue": QUEUE_ID, "type": "ranked", "start": start, "count": 100},
            method="match-ids",
        )
        ids += page
        if len(page) < 100:
            return ids
        start += 100


def unseen(conn: sqlite3.Connection, match_ids: list[str]) -> list[str]:
    match_ids = list(dict.fromkeys(match_ids))
    if not match_ids:
        return []
    placeholders = ",".join("?" * len(match_ids))
    seen = {
        r[0] for r in conn.execute(
            f"SELECT match_id FROM matches WHERE match_id IN ({placeholders}) "
            f"UNION SELECT match_id FROM skipped WHERE match_id IN ({placeholders})",
            (*match_ids, *match_ids),
        )
    }
    return [m for m in match_ids if m not in seen]


def count_tiers(conn: sqlite3.Connection, platform: str, puuids: list[str]) -> tuple[int, int | None]:
    """(Diamond+ players, Emerald players); the Emerald count is None unless that ladder is downloaded."""
    placeholders = ",".join("?" * len(puuids))
    counts = dict(conn.execute(
        f"SELECT tier, COUNT(*) FROM roster WHERE platform = ? AND puuid IN ({placeholders}) GROUP BY tier",
        (platform, *puuids),
    ).fetchall())
    n_diamond_plus = sum(counts.get(t, 0) for t in LADDERS["diamond_plus"])
    return n_diamond_plus, counts.get("EMERALD", 0) if LADDER == "emerald" else None


def store_match(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    n_diamond_plus, n_emerald = count_tiers(conn, row["platform"], json.loads(row["puuids"]))
    row = dict(row, n_diamond_plus=n_diamond_plus, n_emerald=n_emerald, crawled_from=LADDER)
    cols = ", ".join(row)
    conn.execute(f"INSERT OR IGNORE INTO matches ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
    conn.commit()


def store_skip(conn: sqlite3.Connection, match_id: str, reason: str) -> None:
    conn.execute("INSERT OR IGNORE INTO skipped (match_id, reason) VALUES (?, ?)", (match_id, reason))
    conn.commit()


def crawl_player(
    client: RiotClient,
    conn: sqlite3.Connection,
    region: str,
    platform: str,
    puuid: str,
    crawled_until: int | None,
    since: int,
    stats: Stats,
) -> None:
    crawl_started = int(time.time())
    start_time = max(since, (crawled_until or 0) - RECRAWL_OVERLAP_SECONDS)

    try:
        match_ids = list_match_ids(client, region, puuid, start_time)
    except requests.HTTPError as e:
        log.debug("[%s] could not list matches for %s: %s", region, puuid[:12], e)
        match_ids = None

    if match_ids is not None:
        for match_id in unseen(conn, match_ids):
            try:
                row = parse_match(client.get(region, f"/lol/match/v5/matches/{match_id}", method="match"), region)
                store_match(conn, row)
                stats.add(stats.stored, region)
            except SkipMatch as e:
                store_skip(conn, match_id, str(e))
                stats.add(stats.skipped, region)
            except requests.HTTPError as e:
                store_skip(conn, match_id, f"http_{e.response.status_code if e.response is not None else '?'}")
                stats.add(stats.skipped, region)
            except (KeyError, TypeError, ValueError) as e:
                store_skip(conn, match_id, f"parse_{type(e).__name__}")
                stats.add(stats.skipped, region)

    # only mark the player done once all of their new matches are stored
    conn.execute(
        "INSERT OR REPLACE INTO progress (platform, puuid, crawled_until, last_attempt, found) VALUES (?, ?, ?, ?, ?)",
        (platform, puuid, crawl_started if match_ids is not None else crawled_until, crawl_started,
         len(match_ids) if match_ids is not None else None),
    )
    conn.commit()
    stats.add(stats.players, region)


def region_worker(client: RiotClient, region: str, platforms: list[str], since: int, stats: Stats, stop: threading.Event) -> None:
    conn = connect()
    idle_logged = False
    while not stop.is_set():
        try:
            # only crawl platforms whose Diamond+ roster finished loading at least once,
            # so n_diamond_plus is counted against a complete roster
            ready = [p for p in platforms if get_meta(conn, f"roster_refreshed:{p}")]
            batch = next_players(conn, ready) if ready else []
            if not batch:
                if not idle_logged:
                    log.info("[%s] no players due for a crawl (roster loading, or caught up); waiting", region)
                    idle_logged = True
                stop.wait(IDLE_WAIT_SECONDS)
                continue
            idle_logged = False
            for platform, puuid, crawled_until in batch:
                crawl_player(client, conn, region, platform, puuid, crawled_until, since, stats)
        except CollectionStopped:
            break
        except Exception:
            log.error("[%s] region worker error, retrying in 30s:\n%s", region, traceback.format_exc())
            stop.wait(30)
    conn.close()


# -----------------------------
# Commands
# -----------------------------
def setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(LOG_PATH, encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def resolve_since(conn: sqlite3.Connection, since_arg: str | None) -> int:
    if since_arg:
        since = int(datetime.strptime(since_arg, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        set_meta(conn, "since", str(since))
        return since
    stored = get_meta(conn, "since")
    if stored:
        return int(stored)
    since = int((datetime.now(timezone.utc) - timedelta(days=DEFAULT_LOOKBACK_DAYS)).timestamp())
    set_meta(conn, "since", str(since))
    return since


def cmd_collect(args: argparse.Namespace) -> None:
    setup_logging()
    conn = connect()
    init_db(conn)
    since = resolve_since(conn, args.since)
    conn.close()

    stop = threading.Event()
    keys = ApiKeyManager(ENV_PATH, stop)
    client = RiotClient(keys, stop)
    stats = Stats()

    regions = {r: p for r, p in REGIONS.items() if not args.regions or r in args.regions}
    log.info(
        "Collecting %s ranked solo/duo into %s since %s UTC for %s",
        LADDER_LABELS[LADDER], DB_PATH, datetime.fromtimestamp(since, timezone.utc).strftime("%Y-%m-%d"),
        ", ".join(f"{r} ({'/'.join(p)})" for r, p in regions.items()),
    )

    threads = []
    for region, platforms in regions.items():
        for platform in platforms:
            threads.append(threading.Thread(target=roster_worker, args=(client, platform, stop), name=f"roster-{platform}", daemon=True))
        threads.append(threading.Thread(target=region_worker, args=(client, region, platforms, since, stats, stop), name=f"region-{region}", daemon=True))
    for t in threads:
        t.start()

    started = time.time()
    last_status = started
    last_stored: dict[str, int] = {}
    try:
        while not stop.wait(1.0):
            if args.max_minutes and time.time() - started >= args.max_minutes * 60:
                log.info("Reached --max-minutes, stopping")
                break
            if time.time() - last_status >= STATUS_EVERY_SECONDS:
                stored, skipped, players = stats.snapshot()
                elapsed_h = (time.time() - last_status) / 3600
                parts = []
                for region in regions:
                    new = stored.get(region, 0) - last_stored.get(region, 0)
                    parts.append(f"{region}: +{new} ({new / elapsed_h:,.0f}/h)")
                total = sum(stored.values())
                paused = " | KEY EXPIRED, PAUSED" if keys.paused else ""
                log.info("Stored this run: %d | %s | skipped %d | players crawled %d%s",
                         total, " | ".join(parts), sum(skipped.values()), sum(players.values()), paused)
                last_stored, last_status = stored, time.time()
    except KeyboardInterrupt:
        log.info("Ctrl+C received, stopping (finishing in-flight requests)...")
    stop.set()
    for t in threads:
        t.join(timeout=40)
    stored, skipped, _ = stats.snapshot()
    log.info("Stopped. Stored %d matches this run (%d skipped).", sum(stored.values()), sum(skipped.values()))


def cmd_status(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)
    q = lambda sql, *p: pd.read_sql_query(sql, conn, params=p)  # noqa: E731

    print(f"Database: {DB_PATH.resolve()}")
    since = get_meta(conn, "since")
    if since:
        print(f"Collecting since: {datetime.fromtimestamp(int(since), timezone.utc):%Y-%m-%d} UTC")
    print("\nMatches by ladder crawled:")
    print(q("SELECT crawled_from, COUNT(*) AS matches FROM matches GROUP BY crawled_from").to_string(index=False))
    print("\nMatches by platform:")
    print(q("SELECT region, platform, COUNT(*) AS matches, ROUND(AVG(n_diamond_plus), 2) AS avg_diamond_plus, "
            "ROUND(AVG(n_emerald), 2) AS avg_emerald "
            "FROM matches GROUP BY region, platform ORDER BY region, matches DESC").to_string(index=False))
    print("\nMatches by patch:")
    print(q("SELECT patch, COUNT(*) AS matches, MIN(datetime(game_creation, 'unixepoch')) AS first_game, "
            "MAX(datetime(game_creation, 'unixepoch')) AS last_game FROM matches GROUP BY patch").to_string(index=False))
    day_ago = int(time.time()) - 86400
    print(f"\nStored in the last 24h: {q('SELECT COUNT(*) AS n FROM matches WHERE collected_at >= ?', day_ago)['n'][0]}")
    print("\nDiamond+ players per match (filter with export --min-diamond):")
    print(q("SELECT n_diamond_plus, COUNT(*) AS matches FROM matches GROUP BY n_diamond_plus ORDER BY n_diamond_plus DESC").to_string(index=False))
    print("\nSkipped by reason:")
    print(q("SELECT reason, COUNT(*) AS n FROM skipped GROUP BY reason ORDER BY n DESC").to_string(index=False))
    print("\nRoster and crawl progress:")
    print(q(
        "SELECT r.platform, COUNT(*) AS roster_players, COUNT(p.puuid) AS crawled_at_least_once, "
        "datetime(MAX(r.seen_at), 'unixepoch') AS last_roster_refresh "
        "FROM roster r LEFT JOIN progress p ON p.platform = r.platform AND p.puuid = r.puuid "
        "GROUP BY r.platform ORDER BY roster_players DESC"
    ).to_string(index=False))


def cmd_export(args: argparse.Namespace) -> None:
    conn = connect()
    init_db(conn)
    df = pd.read_sql_query(
        f"""
        SELECT match_id, platform AS source_platform, region AS source_region, patch, game_version,
               game_creation, game_duration, n_diamond_plus, n_emerald, crawled_from,
               {", ".join(f"blue_{r}" for r in ROLES)}, {", ".join(f"red_{r}" for r in ROLES)}, blue_win
        FROM matches
        WHERE n_diamond_plus >= ? AND n_diamond_plus + COALESCE(n_emerald, 0) >= ? AND game_duration >= ?
        ORDER BY game_creation
        """,
        conn,
        params=(args.min_diamond, args.min_emerald_plus, MIN_EXPORT_DURATION_SECONDS),
    )
    if args.out:
        out = Path(args.out)
    elif args.min_emerald_plus:
        out = EXPORT_DIR / f"drafts_emerald{args.min_emerald_plus}plus_{len(df)}.csv"
    elif args.min_diamond:
        out = EXPORT_DIR / f"drafts_diamond{args.min_diamond}plus_{len(df)}.csv"
    else:
        out = EXPORT_DIR / f"drafts_all_{len(df)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Exported {len(df)} matches (n_diamond_plus >= {args.min_diamond}, "
          f"Emerald or higher >= {args.min_emerald_plus}, at least {MIN_EXPORT_DURATION_SECONDS // 60} min long) to {out}")


def cmd_pack(args: argparse.Namespace) -> None:
    """Copy games collected since the last pack into a small file to send to the main collector."""
    conn = connect()
    init_db(conn)
    out = Path(args.out) if args.out else EXPORT_DIR / f"pack_{LADDER}_{datetime.now():%Y%m%d_%H%M}.sqlite"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    out.parent.mkdir(parents=True, exist_ok=True)

    # rowids only grow (rows are never deleted), so they mark what was already packed
    tables = ("matches", "skipped")
    after = {t: 0 if args.all else int(get_meta(conn, f"packed_rowid:{t}") or 0) for t in tables}
    upto = {t: conn.execute(f"SELECT COALESCE(MAX(rowid), 0) FROM {t}").fetchone()[0] for t in tables}
    if all(upto[t] <= after[t] for t in tables):
        print("No new games since the last pack.")
        return
    conn.execute("ATTACH DATABASE ? AS pack", (str(out),))
    for t in tables:
        conn.execute(f"CREATE TABLE pack.{t} AS SELECT * FROM main.{t} WHERE rowid > ? AND rowid <= ?", (after[t], upto[t]))
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM pack.matches").fetchone()[0]
    conn.execute("DETACH DATABASE pack")
    for t in tables:
        set_meta(conn, f"packed_rowid:{t}", str(upto[t]))
    print(f"Packed {n} games into {out.resolve()}\nSend this file to whoever runs the main collector.")


def cmd_merge(args: argparse.Namespace) -> None:
    """Add games from a pack (or a whole collector database) made on another machine."""
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"{path} not found")
    conn = connect()
    init_db(conn)
    conn.execute("ATTACH DATABASE ? AS other", (str(path),))
    cols = [r[1] for r in conn.execute("PRAGMA main.table_info(matches)")]
    missing = set(cols) - {r[1] for r in conn.execute("PRAGMA other.table_info(matches)")}
    if missing:
        raise SystemExit(f"{path} is missing columns {sorted(missing)}; it was made by an older version of the collector (git pull, then pack again)")

    col_list = ", ".join(cols)
    before = conn.execute("SELECT COUNT(*) FROM main.matches").fetchone()[0]
    conn.execute(f"INSERT OR IGNORE INTO main.matches ({col_list}) SELECT {col_list} FROM other.matches")
    conn.execute("INSERT OR IGNORE INTO main.skipped (match_id, reason) SELECT match_id, reason FROM other.skipped")
    conn.commit()
    added = conn.execute("SELECT COUNT(*) FROM main.matches").fetchone()[0] - before
    offered = conn.execute("SELECT COUNT(*) FROM other.matches").fetchone()[0]
    print(f"Added {added} of {offered} games from {path} ({offered - added} were already here)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="run the collector until Ctrl+C")
    p.add_argument("--since", help="only matches played after this UTC date (YYYY-MM-DD); remembered across runs")
    p.add_argument("--regions", nargs="*", choices=list(REGIONS), help="limit to these routing regions")
    p.add_argument("--max-minutes", type=float, help="stop automatically after this many minutes")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("status", help="summarize what has been collected")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("export", help="write a CSV for scripts.train_draft_baseline")
    p.add_argument("--min-diamond", type=int, default=0, help="keep matches with at least this many Diamond+ players (0-10); default keeps every game")
    p.add_argument("--min-emerald-plus", type=int, default=0,
                   help="keep matches with at least this many Emerald-or-higher players (0-10); games crawled from "
                        "the Diamond+ ladder have no Emerald count, so only their Diamond+ players count")
    p.add_argument("--out", help="output CSV path")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("pack", help="write games collected since the last pack to a file for `merge`")
    p.add_argument("--all", action="store_true", help="pack every game, not only new ones")
    p.add_argument("--out", help="output .sqlite path")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("merge", help="add games from a pack made by another collector")
    p.add_argument("path", help="pack .sqlite file (or another collector database)")
    p.set_defaults(func=cmd_merge)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
