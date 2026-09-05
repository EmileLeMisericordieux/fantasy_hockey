"""
Assembles the two modelling tables.

There are two different prediction problems in this pool:

  DRAFT     One row per player-season. Inputs are prior seasons only. Answered
            once, in October, and it decides most of the outcome.

  IN-SEASON One row per player-game. Inputs include everything played so far.
            Answered continuously, to drive add/drops and lineup calls.

They need different features, different targets and different validation.
"""

from __future__ import annotations

import json

import pandas as pd

from . import data, features, scoring
from .config import HISTORY_SEASONS, PROCESSED, TARGET_SEASON
from .features import PROVENANCE


def _load(seasons=None, include_goalies=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = list(seasons or HISTORY_SEASONS)
    games = data.load_seasons(seasons, include_goalies=include_goalies)
    games = scoring.add_fantasy_points(games)
    bios = data.player_bios(games["playerId"].unique())
    return games, bios


def inseason_table(seasons=None, include_goalies=True, cache=True) -> pd.DataFrame:
    """Game-level table with rolling form, cumulative state and the remaining-points target."""
    return _cached(
        "inseason", cache, seasons, include_goalies,
        lambda games, bios: features.add_inseason_target(
            features.build_game_features(games, bios)
        ),
    )


def draft_table(seasons=None, include_goalies=True, cache=True) -> pd.DataFrame:
    """Player-season table with prior-season features and this season's output."""
    return _cached("draft", cache, seasons, include_goalies, features.build_draft_table)


def _cached(name: str, cache: bool, seasons, include_goalies, build) -> pd.DataFrame:
    """
    Build a table, or load it from parquet.

    PROVENANCE is populated as a side effect of running the feature builders, so
    a cached load would otherwise come back with some features unregistered and
    silently score differently from a fresh build. The registry is saved beside
    the parquet and restored with it.
    """
    path = PROCESSED / f"{name}.parquet"
    reg_path = PROCESSED / f"{name}_provenance.json"

    if cache and path.exists() and reg_path.exists():
        PROVENANCE.update(json.loads(reg_path.read_text()))
        return pd.read_parquet(path)

    games, bios = _load(seasons, include_goalies)
    df = build(games, bios)
    if cache:
        df.to_parquet(path, index=False)
        reg_path.write_text(json.dumps(
            {c: p for c, p in PROVENANCE.items() if c in df.columns}, indent=1
        ))
    return df


def upcoming_draft_table(target_season: str | None = None,
                         seasons=None, include_goalies=True) -> pd.DataFrame:
    """
    Feature rows for a season that has not been played yet.

    The historical draft table only has rows for seasons with outcomes, so there
    is nothing to score for 2026-27. This synthesises one row per player who
    appeared last season, lags their prior seasons into place, and leaves the
    target empty. Ages advance correctly because they derive from `season_start`.

    Anyone whose NHL career has ended still appears -- box scores cannot tell you
    a player retired or signed in Europe. Filter against the pool's real player
    list before draft day.
    """
    target_season = target_season or TARGET_SEASON
    target_start = int(str(target_season)[:4])

    games, bios = _load(seasons, include_goalies)
    totals = features.add_moneypuck(features.season_totals(games))

    # Everyone who played in the most recent completed season is a candidate.
    latest = totals["season_start"].max()
    roster = (
        totals[totals["season_start"] == latest][["playerId", "positionCode"]]
        .drop_duplicates("playerId")
        .copy()
    )
    roster["season"] = target_season
    roster["season_start"] = target_start

    stacked = pd.concat([totals, roster], ignore_index=True)
    stacked = features.add_prior_season_features(stacked)
    stacked = features.add_position_dummies(stacked)
    stacked = features.add_bio(stacked, bios)

    upcoming = stacked[stacked["season_start"] == target_start].copy()

    names = (
        games.dropna(subset=["playerName"])
        .drop_duplicates("playerId")
        .set_index("playerId")["playerName"]
    )
    upcoming["playerName"] = upcoming["playerId"].map(names)
    return upcoming.reset_index(drop=True)


# ─── Choosing the feature set ──────────────────────────────────────────────────

# Identifiers and raw per-game outcomes. Never inputs.
ID_COLS = {
    "playerId", "playerName", "season", "season_start", "gameId", "gameDate",
    "positionCode", "teamAbbrev", "opponentAbbrev", "commonName",
    "opponentCommonName", "homeRoadFlag", "toi", "decision", "birthDate",
    "shootsCatches", "draft_year", "draft_round", "lastName", "seasonId",
}

RAW_OUTCOME_COLS = {
    "goals", "assists", "points", "shots", "shifts", "pim", "plusMinus",
    "powerPlayGoals", "powerPlayPoints", "shorthandedGoals", "shorthandedPoints",
    "gameWinningGoals", "otGoals", "toi_minutes", "fantasy_points", "points_per60",
    "goalsAgainst", "shotsAgainst", "savePctg", "shutouts", "gamesStarted",
    "season_fantasy_total", "fantasy_remaining", "fp_per_game", "fp_per82",
    "games_played", "toi_per_game", "games_pct", "season_len",
}


def feature_columns(df: pd.DataFrame, task: str = "inseason") -> list[str]:
    """
    Every numeric column available to this task.

    A column has to be registered in features.PROVENANCE to be picked up, so
    adding a feature means saying when it becomes knowable. On draft day that
    is prior seasons only; in-season you also get everything played so far.
    """
    legal = {"draft": {"before_season"}, "inseason": {"before_season", "during_season"}}[task]
    cols = []
    for c in df.columns:
        if c in ID_COLS or c in RAW_OUTCOME_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            continue
        if PROVENANCE.get(c) in legal:
            cols.append(c)
    return sorted(cols)
