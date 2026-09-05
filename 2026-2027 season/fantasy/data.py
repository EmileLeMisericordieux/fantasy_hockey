"""
NHL data acquisition, with a parquet cache per season.

Two things worth knowing:
  * The cache holds RAW api responses, not engineered features, so changing a
    feature never means re-downloading. Features recompute in seconds.
  * Each season is its own file, so the fetch is resumable and a crash costs
    you one season rather than the lot.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import pandas as pd
from nhlpy import NHLClient
from tqdm import tqdm

from .config import (
    API_WORKERS,
    GAME_TYPE_REGULAR,
    PAGE_LIMIT,
    RAW,
)

_client: NHLClient | None = None


def client() -> NHLClient:
    global _client
    if _client is None:
        _client = NHLClient(debug=False)
    return _client


# ─── Season summaries (used to enumerate who played) ───────────────────────────


def _paginate(fetch, **kwargs) -> pd.DataFrame:
    """Walk a paginated NHL stats endpoint until we have every row."""
    rows: list[dict] = []
    start = 0
    while True:
        batch = fetch(start=start, limit=PAGE_LIMIT, **kwargs)
        # nhlpy returns either a bare list or a {'data', 'total'} envelope
        if isinstance(batch, dict):
            data, total = batch.get("data", []), batch.get("total", 0)
        else:
            data, total = batch, None
        rows.extend(data)
        if not data:
            break
        if total is not None and len(rows) >= total:
            break
        if total is None and len(data) < PAGE_LIMIT:
            break
        start += PAGE_LIMIT
    return pd.DataFrame(rows)


def fetch_skater_summary(season: str) -> pd.DataFrame:
    df = _paginate(
        client().stats.skater_stats_summary,
        start_season=season,
        end_season=season,
        game_type_id=GAME_TYPE_REGULAR,
    )
    df["season"] = season
    return df


def fetch_goalie_summary(season: str) -> pd.DataFrame:
    """Goalies come from their own endpoint -- the skater one never returns them."""
    df = _paginate(
        client().stats.goalie_stats_summary,
        start_season=season,
        end_season=season,
        game_type_id=GAME_TYPE_REGULAR,
    )
    if not df.empty:
        df["season"] = season
        df["positionCode"] = "G"
    return df


def season_summary(season: str, include_goalies: bool = True) -> pd.DataFrame:
    """Season-level summary rows for every player who appeared. Cached."""
    path = RAW / f"summary_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    skaters = fetch_skater_summary(season)
    frames = [skaters]
    if include_goalies:
        goalies = fetch_goalie_summary(season)
        if not goalies.empty:
            frames.append(goalies)

    df = pd.concat(frames, ignore_index=True)
    if not df.empty:
        # Never cache an empty result. A season that has not started yet returns
        # nothing, and caching that would keep returning nothing after it starts.
        df.to_parquet(path, index=False)
    return df


# ─── Game logs ─────────────────────────────────────────────────────────────────


def _fetch_one_log(player_id: int, season: str, position: str) -> list[dict]:
    try:
        logs = client().stats.player_game_log(
            player_id=str(player_id), season_id=season, game_type=GAME_TYPE_REGULAR
        )
    except Exception:
        return []
    for g in logs:
        g["playerId"] = player_id
        g["season"] = season
        g["positionCode"] = position
    return logs


def game_logs(season: str, include_goalies: bool = True) -> pd.DataFrame:
    """
    One row per player-game for a season. Cached to data/raw/games_<season>.parquet.
    """
    path = RAW / f"games_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    summary = season_summary(season, include_goalies=include_goalies)
    if summary.empty:
        return pd.DataFrame()

    combos = (
        summary[["playerId", "positionCode"]]
        .drop_duplicates("playerId")
        .reset_index(drop=True)
    )

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        futures = [
            pool.submit(_fetch_one_log, int(r.playerId), season, r.positionCode)
            for r in combos.itertuples(index=False)
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"logs {season}"):
            rows.extend(f.result())

    df = pd.DataFrame(rows)
    if df.empty:
        return df   # not cached on purpose -- see season_summary

    # Attach a single canonical name column. Order matters: lastName is only a
    # fallback, so it is filled first and the full-name columns overwrite it.
    names: dict = {}
    for col in ("lastName", "goalieFullName", "skaterFullName"):
        if col in summary.columns:
            names.update(
                summary.dropna(subset=[col]).set_index("playerId")[col].to_dict()
            )
    df["playerName"] = df["playerId"].map(names)

    df["toi_minutes"] = parse_toi(df["toi"]) if "toi" in df.columns else 0.0
    df.to_parquet(path, index=False)
    return df


def parse_toi(series: pd.Series) -> pd.Series:
    """'18:42' -> 18.7 minutes. Vectorised; non-parsing values become 0."""
    s = series.astype(str)
    parts = s.str.split(":", n=1, expand=True)
    mins = pd.to_numeric(parts[0], errors="coerce")
    secs = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else 0
    return (mins + pd.Series(secs, index=series.index).fillna(0) / 60).fillna(0.0)


def load_seasons(seasons: Iterable[str], include_goalies: bool = True) -> pd.DataFrame:
    """Concatenated game logs for several seasons."""
    frames = [game_logs(s, include_goalies=include_goalies) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["gameDate"] = pd.to_datetime(df["gameDate"])
    return df


# ─── Player biography ──────────────────────────────────────────────────────────


def player_bios(player_ids: Iterable[int]) -> pd.DataFrame:
    """
    Height, weight, birth date, draft position. Cached across all seasons in one
    file, since biography does not change.
    """
    path = RAW / "player_bios.parquet"
    known = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["playerId"])

    wanted = set(int(p) for p in player_ids)
    missing = sorted(wanted - set(known["playerId"].astype(int)) if len(known) else wanted)

    if missing:
        def _one(pid: int) -> dict:
            try:
                info = client().stats.player_career_stats(player_id=str(pid))
            except Exception:
                return {"playerId": pid}
            draft = info.get("draftDetails") or {}
            return {
                "playerId": pid,
                "height_cm": info.get("heightInCentimeters"),
                "weight_kg": info.get("weightInKilograms"),
                "birthDate": info.get("birthDate"),
                "shootsCatches": info.get("shootsCatches"),
                "draft_year": draft.get("year"),
                "draft_round": draft.get("round"),
                "overall_pick": draft.get("overallPick"),
            }

        rows = []
        with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
            futures = [pool.submit(_one, pid) for pid in missing]
            for f in tqdm(as_completed(futures), total=len(futures), desc="bios"):
                rows.append(f.result())

        known = pd.concat([known, pd.DataFrame(rows)], ignore_index=True)
        known = known.drop_duplicates("playerId", keep="last")
        known.to_parquet(path, index=False)

    return known
