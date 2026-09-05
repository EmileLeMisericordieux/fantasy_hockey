"""
Feature builders -- the columns we feed the model.

Every feature is tagged with *when you would know it*, because the two models
know different things:

    "before_season"  Known before the season starts: age, draft position, last
                     year's stats. Both models can use these.
    "during_season"  Only known once games have been played: current form,
                     points so far. The in-season model can use these; the
                     draft model cannot, because on draft day none of it
                     has happened yet.
    "outcome"        What we are trying to predict. A target, never an input.

`datasets.feature_columns()` uses these tags to hand each model the right
columns. Tagging a new feature is what makes it available -- untagged columns
are simply ignored.

To add a feature: compute it, then call `register("my_column", "before_season")`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import moneypuck
from .config import DEFENCE_POS, GOALIE_POS, season_length

# column name -> when you would know it
PROVENANCE: dict[str, str] = {}


def register(cols, when_known: str) -> None:
    """Tag a column with when it becomes knowable. See the module docstring."""
    for c in cols if isinstance(cols, (list, tuple, set)) else [cols]:
        PROVENANCE[c] = when_known


# ─── In-season features (one row per player-game) ──────────────────────────────

ROLL_STATS = [
    "fantasy_points", "goals", "assists", "shots", "toi_minutes",
    "plusMinus", "points_per60", "powerPlayPoints",
]


def add_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-60 production. Normalises output by ice time."""
    df = df.copy()
    if {"goals", "assists", "toi_minutes"}.issubset(df.columns):
        df["points_per60"] = (df["goals"] + df["assists"]) / df["toi_minutes"].clip(lower=1) * 60
    return df


def add_rolling(df: pd.DataFrame, windows=(3, 5, 10, 20)) -> pd.DataFrame:
    """
    Rolling means of recent form, shifted by one game so a row never sees its
    own result.
    """
    df = df.sort_values(["playerId", "gameDate"]).copy()
    cols = [c for c in ROLL_STATS if c in df.columns]

    grouped = df.groupby("playerId", sort=False)
    for w in windows:
        for col in cols:
            name = f"roll{w}_{col}"
            df[name] = grouped[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )
            register(name, "during_season")
    return df


def add_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date state, all shifted so the current game is excluded."""
    df = df.sort_values(["playerId", "season", "gameDate"]).copy()
    g = df.groupby(["playerId", "season"], sort=False)

    df["games_so_far"] = g.cumcount()
    df["cum_fantasy_so_far"] = g["fantasy_points"].transform(
        lambda s: s.shift(1).cumsum()
    ).fillna(0)

    length = df["season"].map(season_length)
    df["season_length"] = length
    df["team_games_remaining"] = (length - df["games_so_far"]).clip(lower=0)
    df["season_progress"] = (df["games_so_far"] / length).clip(upper=1.0)
    df["fantasy_per_game_so_far"] = df["cum_fantasy_so_far"] / df["games_so_far"].clip(lower=1)
    df["pace_projection"] = df["fantasy_per_game_so_far"] * length

    register(
        ["games_so_far", "cum_fantasy_so_far", "season_length", "team_games_remaining",
         "season_progress", "fantasy_per_game_so_far", "pace_projection"],
        "during_season",
    )
    return df


def add_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Days of rest and home/away. Both known before puck drop."""
    df = df.sort_values(["playerId", "gameDate"]).copy()
    df["days_rest"] = (
        df.groupby("playerId", sort=False)["gameDate"].diff().dt.days.fillna(7).clip(upper=15)
    )
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)
    if "homeRoadFlag" in df.columns:
        df["is_home"] = (df["homeRoadFlag"] == "H").astype(int)
        register("is_home", "before_season")
    register(["days_rest", "is_back_to_back"], "before_season")
    return df


def add_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot position, plus the grouping the scoring rules actually care about.

    A defence goal is worth 3 and an assist 2, versus 2 and 1 for a forward, so
    `is_defence` is doing real work here -- it is not just a nuisance category.
    """
    df = df.copy()
    for pos in ("C", "L", "R", "D", "G"):
        name = f"pos_{pos}"
        df[name] = (df["positionCode"] == pos).astype(int)
        register(name, "before_season")
    df["is_defence"] = df["positionCode"].isin(DEFENCE_POS).astype(int)
    df["is_goalie"] = df["positionCode"].isin(GOALIE_POS).astype(int)
    register(["is_defence", "is_goalie"], "before_season")
    return df


def add_bio(df: pd.DataFrame, bios: pd.DataFrame) -> pd.DataFrame:
    """Age, size, draft pedigree. Fixed before the season starts."""
    df = df.merge(bios, on="playerId", how="left")

    season_start = df["season"].astype(str).str[:4].astype(int)
    birth_year = pd.to_datetime(df["birthDate"], errors="coerce").dt.year
    df["age"] = season_start - birth_year
    df["age_sq"] = df["age"] ** 2
    df["draft_age"] = pd.to_numeric(df["draft_year"], errors="coerce") - birth_year
    df["is_undrafted"] = df["overall_pick"].isna().astype(int)
    df["overall_pick"] = pd.to_numeric(df["overall_pick"], errors="coerce").fillna(250)

    register(
        ["age", "age_sq", "draft_age", "is_undrafted", "overall_pick",
         "height_cm", "weight_kg"],
        "before_season",
    )
    return df


def build_game_features(df: pd.DataFrame, bios: pd.DataFrame) -> pd.DataFrame:
    """The full in-season feature pipeline."""
    df = add_rate_stats(df)
    df = add_rolling(df)
    df = add_cumulative(df)
    df = add_rest(df)
    df = add_position_dummies(df)
    df = add_bio(df, bios)
    return df


# ─── The target for the in-season model ────────────────────────────────────────


def add_inseason_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    season_fantasy_total : full-season sum
    fantasy_remaining    : points still to come after this game  <- the target
    """
    df = df.copy()
    df["season_fantasy_total"] = df.groupby(["playerId", "season"])["fantasy_points"].transform("sum")
    df["fantasy_remaining"] = (df["season_fantasy_total"] - df["cum_fantasy_so_far"]).clip(lower=0)
    register(["season_fantasy_total", "fantasy_remaining"], "outcome")
    return df


# ─── Draft-model features (one row per player-season) ──────────────────────────


def season_totals(games: pd.DataFrame) -> pd.DataFrame:
    """Collapse game logs to one row per player-season."""
    agg = {
        "fantasy_points": "sum",
        "goals": "sum",
        "assists": "sum",
        "shots": "sum",
        "toi_minutes": "sum",
        "plusMinus": "sum",
        "gameDate": "count",
    }
    agg = {k: v for k, v in agg.items() if k in games.columns}
    if "powerPlayPoints" in games.columns:
        agg["powerPlayPoints"] = "sum"

    out = (
        games.groupby(["playerId", "season", "positionCode"], as_index=False)
        .agg(agg)
        .rename(columns={"gameDate": "games_played"})
    )
    out["season_start"] = out["season"].astype(str).str[:4].astype(int)
    length = out["season"].map(season_length)
    out["season_len"] = length

    # fp = fantasy points. fp_per82 is "what this player would have scored over
    # a full 82-game season", which makes the two short COVID seasons
    # comparable with the rest.
    out["fp_per_game"] = out["fantasy_points"] / out["games_played"].clip(lower=1)
    out["fp_per82"] = out["fp_per_game"] * 82
    out["games_pct"] = out["games_played"] / length
    if "toi_minutes" in out.columns:
        out["toi_per_game"] = out["toi_minutes"] / out["games_played"].clip(lower=1)
    return out.sort_values(["playerId", "season_start"])


def add_moneypuck(totals: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Attach MoneyPuck's advanced stats to the player-season table.

    Expected goals, shot danger, on-ice rates and situational ice time -- the
    power-play split above all, which the NHL API does not expose at all. See
    `fantasy/moneypuck.py`; it joins on `playerId` + `season` because MoneyPuck
    uses NHL player ids.

    **Nothing is registered here on purpose.** These columns are the season's
    own numbers, and handing the draft model those would be handing it the
    answer. `add_prior_season_features` lags every numeric `mp_*` column into
    `prev1_mp_*`, `prev2_mp_*`, `prev3_mp_*`, and *those* are the features.

    A no-op if the data has not been downloaded -- run
    `python scripts/fetch_moneypuck.py`.
    """
    mp = moneypuck.player_season_table() if table is None else table
    if mp.empty:
        return totals

    out = totals.copy()
    out["playerId"] = out["playerId"].astype("int64")
    out["season"] = out["season"].astype(str)
    return out.merge(mp, on=["playerId", "season"], how="left")


def add_prior_season_features(totals: pd.DataFrame, n_back: int = 3) -> pd.DataFrame:
    """
    For each player-season, attach what was known from previous seasons.

    This is the draft model's entire world: on draft day you have last year's
    numbers, the two before that, and a birth certificate.
    """
    df = totals.sort_values(["playerId", "season_start"]).copy()
    g = df.groupby("playerId", sort=False)

    carry = [c for c in ["fp_per82", "fp_per_game", "games_played", "games_pct",
                         "goals", "assists", "shots", "toi_per_game", "powerPlayPoints"]
             if c in df.columns]

    # Every MoneyPuck column gets the same treatment, if `add_moneypuck` ran.
    # They are prior-season facts exactly like the rest of this list, so they
    # lag the same way -- and lagging is what makes them legal to use.
    carry += sorted(c for c in df.columns
                    if c.startswith("mp_") and pd.api.types.is_numeric_dtype(df[c]))

    # Built as one block and concatenated once. Inserting a few hundred columns
    # one at a time fragments the frame badly enough that pandas warns about it.
    lagged = {}
    for lag in range(1, n_back + 1):
        for col in carry:
            name = f"prev{lag}_{col}"
            lagged[name] = g[col].shift(lag)
            register(name, "before_season")
    df = pd.concat([df, pd.DataFrame(lagged, index=df.index)], axis=1)

    # Weighted average of the last 3 seasons, where recent seasons count more
    # (last season weight 5, the one before 4, the one before that 3). Players
    # with fewer than 3 seasons just use what they have.
    weights = [5, 4, 3][:n_back]
    smoothed = ["fp_per82", "games_pct"]
    # Role and underlying rates are noisy in any single season -- a three-year
    # blend of "how much does this player play, and how good are his chances"
    # is steadier than last year alone.
    smoothed += [c for c in ("mp_toi_per_game", "mp_pp_toi_per_game",
                             "mp_pp_toi_share_of_team", "mp_xgoals_per60",
                             "mp_points_per60", "mp_onice_xg_pct",
                             "mp_g_gsax_per60") if c in df.columns]
    for col in smoothed:
        cols = [f"prev{lag}_{col}" for lag in range(1, n_back + 1) if f"prev{lag}_{col}" in df]
        if not cols:
            continue
        vals = df[cols].to_numpy(dtype=float)
        w = np.array(weights[: len(cols)], dtype=float)
        mask = ~np.isnan(vals)
        wsum = (mask * w).sum(axis=1)
        num = np.nansum(vals * w * mask, axis=1)
        name = f"avg3_{col}"
        df[name] = np.where(wsum > 0, num / np.maximum(wsum, 1e-9), np.nan)
        register(name, "before_season")

    # Trajectory: is this player trending up or down?
    if {"prev1_fp_per82", "prev2_fp_per82"}.issubset(df.columns):
        df["trend_fp_per82"] = df["prev1_fp_per82"] - df["prev2_fp_per82"]
        register("trend_fp_per82", "before_season")

    df["seasons_of_history"] = g.cumcount()
    register("seasons_of_history", "before_season")
    return df


def build_draft_table(games: pd.DataFrame, bios: pd.DataFrame) -> pd.DataFrame:
    """
    One row per player-season, with pre-season features and the season's actual
    fantasy output as the target.
    """
    totals = season_totals(games)
    totals = add_moneypuck(totals)
    df = add_prior_season_features(totals)
    df = add_position_dummies(df)
    df = add_bio(df, bios)

    # Targets. fp_per82 is the fairer one to model across short seasons;
    # fantasy_points is what the pool actually pays out.
    register(["fantasy_points", "fp_per82", "fp_per_game", "games_played"], "outcome")
    return df
