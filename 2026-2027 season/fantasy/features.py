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

from . import moneypuck, teams
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

    # How much of the season is left by the *calendar*, not by the player's own
    # game count. `team_games_remaining` above counts 82 minus the games he has
    # played, so a player who missed twenty is handed twenty nights that do not
    # exist — and it hands them to exactly the players least likely to be
    # there for them. Anything projecting a season total needs this one instead.
    by_season = df.groupby("season")["gameDate"]
    opening = by_season.transform("min")
    span = (by_season.transform("max") - opening).dt.days.clip(lower=1)
    df["season_elapsed_pct"] = ((df["gameDate"] - opening).dt.days / span).clip(0, 1)
    df["team_games_left_est"] = length * (1 - df["season_elapsed_pct"])

    register(
        ["games_so_far", "cum_fantasy_so_far", "season_length", "team_games_remaining",
         "season_progress", "fantasy_per_game_so_far", "pace_projection",
         "season_elapsed_pct", "team_games_left_est"],
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


def add_prior_season_context(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Give every game row what was already known about the player on opening
    night: last season's rates, the MoneyPuck advanced stats, the three-year
    blends.

    Without this the in-season model is blind early. After five games a pace
    projection is almost pure noise, and the only real information about a
    player is what he did last year — which is exactly what the draft model
    runs on.

    Built by running the draft table's own prior-season pipeline and merging
    the result onto game rows by player and season, so the two models see
    identical numbers and a fix to one fixes both. Everything merged is already
    tagged `before_season`, which is what makes it legal here.
    """
    totals = add_moneypuck(season_totals(games))
    prior = add_prior_season_features(totals)

    keep = [c for c in prior.columns
            if c.startswith(("prev", "avg3_", "trend_")) or c == "seasons_of_history"]
    prior = prior[["playerId", "season"] + keep].copy()
    prior["playerId"] = prior["playerId"].astype("int64")
    prior["season"] = prior["season"].astype(str)

    out = df.copy()
    out["playerId"] = out["playerId"].astype("int64")
    out["season"] = out["season"].astype(str)
    # Anything already built at game level wins; these are the season-level view.
    prior = prior.drop(columns=[c for c in keep if c in out.columns])
    return out.merge(prior, on=["playerId", "season"], how="left")


def add_moneypuck_game_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge MoneyPuck's per-game rows on. Two things arrive that the NHL game log
    does not have at all: **ice time on the power play in a single game**, and
    **expected goals in a single game**.

    Joins on player and `gameId`, which is the NHL game id in both. Verified on
    2024-25: 47,217 of 47,217 skater-games matched, team codes agreed on every
    one, and total ice time was identical to three decimal places.

    Nothing is registered here — these are the game's own numbers. The builders
    below turn them into shifted rolling features, which is what a row is
    allowed to know.
    """
    mpg = moneypuck.load_game_logs()
    if mpg.empty:
        return df

    cols = ["playerId", "gameId", "toi_pp", "toi_ev", "toi_pk",
            "xg_all", "xg_pp", "shots_all", "game_score_all", "onice_xg_pct_ev"]
    mpg = mpg[[c for c in cols if c in mpg.columns]].copy()
    mpg["playerId"] = mpg["playerId"].astype("int64")
    mpg["gameId"] = mpg["gameId"].astype("int64")

    out = df.copy()
    out["playerId"] = out["playerId"].astype("int64")
    out["gameId"] = out["gameId"].astype("int64")
    return out.merge(mpg, on=["playerId", "gameId"], how="left")


def _by_player_season(df: pd.DataFrame):
    """Grouper for within-season history. Never carries March into October."""
    return df.groupby(["playerId", "season"], sort=False)


def add_pp_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Power-play deployment, and — the point of it — whether it is *moving*.

    `pp_toi_delta5` is the mean power-play ice time over the last five games
    minus the mean over the five before that. Positive means a player has been
    promoted; a jump from PP2 to PP1 is one of the largest fantasy swings there
    is, and it shows up here weeks before the points do.

    Grouped within a season, so a player's last five games of April never leak
    into his first five of October.
    """
    df = df.sort_values(["playerId", "season", "gameDate"]).copy()
    if "toi_pp" not in df.columns:
        return df
    g = _by_player_season(df)

    df["roll5_pp_toi"] = g["toi_pp"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["roll20_pp_toi"] = g["toi_pp"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    prev5 = g["toi_pp"].transform(
        lambda s: s.shift(6).rolling(5, min_periods=1).mean())
    df["pp_toi_delta5"] = df["roll5_pp_toi"] - prev5

    roll5_toi = g["toi_minutes"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["pp_toi_share5"] = df["roll5_pp_toi"] / roll5_toi.clip(lower=0.1)

    register(["roll5_pp_toi", "roll20_pp_toi", "pp_toi_delta5", "pp_toi_share5"],
             "during_season")
    return df


def add_expected_goals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected goals: how many a player *should* have scored from the shots he
    took, given where and how he took them.

    The reason to want it is that it settles down far faster than goals do. Ten
    games of goals is mostly luck; ten games of expected goals is mostly the
    player. `xg_diff20` is the gap between the two — a player far above his
    expected goals has been finishing hot and is due to come back.
    """
    df = df.sort_values(["playerId", "season", "gameDate"]).copy()
    if "xg_all" not in df.columns:
        return df
    g = _by_player_season(df)

    for w in (5, 10, 20):
        df["roll%d_xg" % w] = g["xg_all"].transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
        register("roll%d_xg" % w, "during_season")

    df["roll20_xg_pp"] = g["xg_pp"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    roll20_goals = g["goals"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    df["xg_diff20"] = roll20_goals - df["roll20_xg"]

    register(["roll20_xg_pp", "xg_diff20"], "during_season")
    return df


def add_team_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Whether the player has moved. A trade resets everything around him — new
    linemates, new power-play unit, new coach — so his own recent history is a
    weaker guide than usual right after one.

    `changed_team` is within this season (traded midway).
    `changed_team_offseason` compares against the last team he played for in
    the previous season.
    """
    df = df.sort_values(["playerId", "season", "gameDate"]).copy()
    g = _by_player_season(df)

    df["changed_team"] = (df["teamAbbrev"] != g["teamAbbrev"].transform("first")).astype(int)
    df["n_teams_so_far"] = g["teamAbbrev"].transform(
        lambda s: (~s.duplicated()).cumsum()).astype(float)

    # Last team of the player's previous season.
    last = (df.sort_values("gameDate")
              .groupby(["playerId", "season"])["teamAbbrev"].last()
              .reset_index().rename(columns={"teamAbbrev": "_last_team"}))
    last["_ss"] = last["season"].astype(str).str[:4].astype(int)
    last = last.sort_values(["playerId", "_ss"])
    last["_prev_team"] = last.groupby("playerId")["_last_team"].shift(1)

    df = df.merge(last[["playerId", "season", "_prev_team"]],
                  on=["playerId", "season"], how="left")
    df["changed_team_offseason"] = (
        df["_prev_team"].notna() & (df["teamAbbrev"] != df["_prev_team"])
    ).astype(int)
    df = df.drop(columns=["_prev_team"])

    register(["changed_team", "n_teams_so_far", "changed_team_offseason"], "during_season")
    return df


def add_team_context(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Where the player's team stands, and what it still has to play.

    Built by `fantasy/teams.py` out of the game logs already on disk — team
    results come off the goalie rows. Everything is as of the morning of the
    game, and the remaining schedule is legitimate to know because the NHL
    publishes it in advance.

    `team_games_left` here is the **exact** count from the schedule and
    replaces the calendar approximation `team_games_left_est`, which is kept
    only as a fallback for rows the merge cannot reach.
    """
    ctx = teams.team_context(games)
    out = df.merge(
        ctx, how="left",
        left_on=["season", "teamAbbrev", "gameDate"],
        right_on=["season", "team", "gameDate"],
    ).drop(columns=["team"], errors="ignore")

    if "team_games_left_est" in out.columns:
        out["team_games_left"] = out["team_games_left"].fillna(out["team_games_left_est"])

    register(
        ["team_rank", "team_points", "team_points_pct", "team_games_played",
         "team_gf_per_game", "team_ga_per_game", "team_goal_diff_per_game",
         "team_games_left", "opp_ga_per_game_left", "opp_goal_diff_left",
         "games_left_vs_weak", "pct_left_vs_weak"],
        "during_season",
    )
    return out


def add_availability(df: pd.DataFrame) -> pd.DataFrame:
    """
    How many of the remaining games this player will actually be there for, and
    the closest thing to injury data we have.

    **There is no injury feed here.** Nothing in this repo knows a player is
    hurt. What it can see is that his team kept playing and he did not, which is
    the same fact arriving a few days late. `games_missed_so_far` is exactly
    that gap, and `days_since_last_game` catches a player on his way back.

    `expected_games_left` is the games half of "season total = rate x games":
    the schedule's remaining games, scaled by how available this player has
    actually been. Early in the year that is mostly last season's record,
    because twelve games say very little; by March it is mostly this season's.
    """
    df = df.sort_values(["playerId", "season", "gameDate"]).copy()
    g = _by_player_season(df)

    df["days_since_last_game"] = g["gameDate"].diff().dt.days
    df["is_returning"] = (df["days_since_last_game"] > 7).astype(float)
    register(["days_since_last_game", "is_returning"], "during_season")

    if "team_games_played" not in df.columns:
        return df

    team_gp = df["team_games_played"].fillna(df["games_so_far"])
    df["games_missed_so_far"] = (team_gp - df["games_so_far"]).clip(lower=0)
    df["availability_so_far"] = df["games_so_far"] / team_gp.clip(lower=1)

    # Games the player sat out across his own last ten appearances.
    team_gp_10ago = g["team_games_played"].transform(lambda s: s.shift(10))
    df["games_missed_last10"] = (team_gp - team_gp_10ago - 10).clip(lower=0)

    # Shrink this season's availability toward last season's, by sample size.
    k = 20.0
    w = team_gp / (team_gp + k)
    prior = df["prev1_games_pct"] if "prev1_games_pct" in df.columns else pd.Series(np.nan, index=df.index)
    prior = prior.fillna(df["availability_so_far"])
    df["expected_availability"] = w * df["availability_so_far"] + (1 - w) * prior
    df["expected_games_left"] = df["team_games_left"] * df["expected_availability"]

    if "roll20_xg" in df.columns:
        df["expected_goals_left"] = df["roll20_xg"] * df["expected_games_left"]
        register("expected_goals_left", "during_season")

    register(
        ["games_missed_so_far", "availability_so_far", "games_missed_last10",
         "expected_availability", "expected_games_left"],
        "during_season",
    )
    return df


def build_game_features(df: pd.DataFrame, bios: pd.DataFrame,
                        games: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    The full in-season feature pipeline.

    Order matters in three places:
      * MoneyPuck's per-game rows land before the rolling builders, because
        power-play ice time and expected goals are what they roll over.
      * Team context lands before availability, which needs to know how many
        games the *team* has played to work out how many the player missed.
      * Prior-season context lands before availability too, which shrinks this
        season's availability toward last season's.
    """
    games = games if games is not None else df
    df = add_rate_stats(df)
    df = add_moneypuck_game_stats(df)
    df = add_rolling(df)
    df = add_cumulative(df)
    df = add_rest(df)
    df = add_position_dummies(df)
    df = add_bio(df, bios)
    df = add_prior_season_context(df, games)
    df = add_team_context(df, games)
    df = add_pp_usage(df)
    df = add_expected_goals(df)
    df = add_team_change(df)
    df = add_availability(df)
    # Split key for walk-forward validation. In ID_COLS, so never a feature.
    df["season_start"] = df["season"].astype(str).str[:4].astype(int)
    return df


# ─── The target for the in-season model ────────────────────────────────────────


def add_inseason_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    season_fantasy_total : full-season sum
    fantasy_remaining    : points from this game onward  <- the target

    Onward *including* this game, not after it. Every feature on the row is
    shifted to exclude the row's own game, so the pairing is "everything known
    before puck drop" against "everything scored from puck drop to the end of
    the season" — the question you are answering the morning you set a
    lineup or make a claim.
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
