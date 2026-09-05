"""
MoneyPuck advanced stats, with a parquet cache per season.

MoneyPuck (moneypuck.com) publishes the things the NHL API does not: an
expected-goals model, shot-danger buckets, on-ice rates, and -- the one worth
the most here -- ice time split by situation, so power-play deployment stops
being a guess.

Two facts make this cheap to use:

  * **MoneyPuck's `playerId` IS the NHL player id.** Verified: all 920 skaters
    in 2024-25 match the NHL summary exactly, both directions. Everything joins
    on `playerId` + `season`, no name matching, no crosswalk table.
  * **One row per player per season, not per team.** Traded players are already
    consolidated -- full-season totals, credited to the team they finished the
    season with (Rantanen's 2024-25 row is all 82 games, filed under DAL). So
    there are no duplicates to collapse, but `mp_team` is an end-of-season
    team, and the team-relative columns rank a traded player against the
    teammates he ended up with rather than the ones he played most of the year
    with. Rare, and the right team to know for projecting the season ahead.

Same cache philosophy as `data.py`: `data/raw/moneypuck_<kind>_<season>.parquet`
holds the CSV as published -- all 154 skater columns, all five situations.
Curating features never means re-downloading.

Rows are one per player *per situation*:

    all    every situation combined
    5on5   even strength
    5on4   power play        <- the fantasy-relevant one
    4on5   penalty kill
    other  everything else (4on4, 3on3, empty net, ...)

`all` is the exact sum of the other four. `games_played` repeats on every row.

Units: `icetime` is in **seconds**. Counting stats are season totals.

Season ids follow this repo ("20242025"); MoneyPuck numbers seasons by start
year (2024), which `_mp_year()` converts. Coverage is 2008-09 onward, so every
season in `HISTORY_SEASONS` is available.
"""

from __future__ import annotations

import io
from typing import Iterable

import pandas as pd
import requests

from .config import HISTORY_SEASONS, RAW

BASE_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/{kind}.csv"

KINDS = ("skaters", "goalies")

# MoneyPuck serves plain files, but a bare urllib/requests default agent gets
# inconsistent treatment from the CDN. A normal browser agent does not.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fantasy-hockey/1.0)"}

_TIMEOUT = 120


def _mp_year(season: str | int) -> int:
    """'20242025' -> 2024. MoneyPuck names a season by the year it starts."""
    return int(str(season)[:4])


# ─── Download and cache ────────────────────────────────────────────────────────


def fetch(kind: str, season: str) -> pd.DataFrame:
    """
    Download one season of MoneyPuck data. Not cached -- see `season_stats`.

    Returns an empty frame for a season MoneyPuck has not published yet, which
    is what a 404 means here (the upcoming season has no file until games are
    played).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    url = BASE_URL.format(year=_mp_year(season), kind=kind)
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    if resp.status_code == 404:
        return pd.DataFrame()
    resp.raise_for_status()

    df = pd.read_csv(io.BytesIO(resp.content))
    # Normalise the key: MoneyPuck stores 2024, we key everything on "20242025".
    df["season"] = str(season)
    return df


def season_stats(kind: str, season: str, refresh: bool = False) -> pd.DataFrame:
    """
    One season of MoneyPuck rows, cached to parquet. The full CSV, untouched
    apart from the `season` key.

    `refresh=True` re-downloads. Worth it for a season still in progress --
    MoneyPuck updates nightly and a cached file from October would quietly
    stay wrong all year.
    """
    path = RAW / f"moneypuck_{kind}_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = fetch(kind, season)
    if not df.empty:
        # Never cache an empty result -- see the same rule in data.py.
        df.to_parquet(path, index=False)
    return df


def load_seasons(kind: str, seasons: Iterable[str] | None = None,
                 refresh: bool = False) -> pd.DataFrame:
    """Concatenated MoneyPuck rows for several seasons."""
    seasons = list(seasons or HISTORY_SEASONS)
    frames = [season_stats(kind, s, refresh=refresh) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─── Situational reshaping ─────────────────────────────────────────────────────


def _by_situation(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split the long form into one frame per situation, indexed by
    (playerId, season) so the situations line up for arithmetic.
    """
    out: dict[str, pd.DataFrame] = {}
    for sit, sub in df.groupby("situation", sort=False):
        sub = sub.set_index(["playerId", "season"])
        out[str(sit)] = sub[~sub.index.duplicated(keep="first")]
    return out


def _per60(count: pd.Series, icetime_seconds: pd.Series) -> pd.Series:
    """
    Rate per 60 minutes of ice time.

    Zero ice time gives NaN, not zero: a player who never took a power-play
    shift has an *undefined* power-play rate, not a rate of nothing. The models
    treat NaN as missing, which is the honest answer. Volume is carried
    separately by the `*_toi_per_game` columns, where zero really does mean zero.
    """
    hours = icetime_seconds / 3600.0
    return count / hours.where(hours > 0)


def _share(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN where the denominator is zero."""
    return num / den.where(den > 0)


# ─── Curated per-player-season features ────────────────────────────────────────


def skater_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the five situational rows into one row per player-season of
    model-ready `mp_*` columns.

    Everything here is a **rate, share or per-game** figure. Season totals are
    deliberately left out: the NHL game logs already carry those, and a total
    conflates "how good" with "how many games", which the draft table splits
    apart on purpose.
    """
    if df.empty:
        return pd.DataFrame()

    sit = _by_situation(df)
    a = sit["all"]
    idx = a.index
    ev = sit.get("5on5", pd.DataFrame()).reindex(idx)
    pp = sit.get("5on4", pd.DataFrame()).reindex(idx)
    pk = sit.get("4on5", pd.DataFrame()).reindex(idx)

    gp = a["games_played"].clip(lower=1)
    ice, ice_ev, ice_pp, ice_pk = a["icetime"], ev["icetime"], pp["icetime"], pk["icetime"]

    out = pd.DataFrame(index=idx)
    out["mp_team"] = a["team"]
    out["mp_games_played"] = a["games_played"]

    # ── Opportunity. Ice time is the best single read on what a coach thinks,
    #    and the power-play split is the piece the NHL API cannot give us.
    out["mp_toi_per_game"] = ice / 60 / gp
    out["mp_ev_toi_per_game"] = ice_ev / 60 / gp
    out["mp_pp_toi_per_game"] = ice_pp / 60 / gp
    out["mp_pk_toi_per_game"] = ice_pk / 60 / gp
    out["mp_pp_toi_share"] = _share(ice_pp, ice)
    out["mp_shifts_per_game"] = a["I_F_shifts"] / gp
    zone_starts = a["I_F_oZoneShiftStarts"] + a["I_F_dZoneShiftStarts"]
    out["mp_ozone_start_pct"] = _share(a["I_F_oZoneShiftStarts"], zone_starts)

    # ── Role relative to teammates. Minutes say how much a player plays; his
    #    rank among his own team's forwards or defencemen says whether he is
    #    *the guy*, which is what carries over when the roster around him
    #    changes.
    #
    #    `mp_pp_toi_share_of_team` is the PP1 detector. A team plays five
    #    skaters on the power play, so its total power-play time is the sum of
    #    its players' 5on4 ice time divided by five; a player's share of that is
    #    the fraction of his team's power play he is actually on the ice for.
    #    PP1 regulars land near 0.7, PP2 near 0.3.
    #
    #    MoneyPuck ships its own `iceTimeRank`, which is deliberately not used:
    #    it correlates only 0.33 with ice time and repeats within a team, so
    #    whatever it ranks is not this.
    role = pd.DataFrame(
        {
            "season_key": idx.get_level_values("season"),
            "team": a["team"].to_numpy(),
            "is_d": a["position"].eq("D").to_numpy(),
            "toi": out["mp_toi_per_game"].to_numpy(),
            "pp_toi": out["mp_pp_toi_per_game"].to_numpy(),
            "pp_ice": ice_pp.to_numpy(),
        },
        index=idx,
    )
    by_unit = role.groupby(["season_key", "team", "is_d"], sort=False)
    out["mp_toi_rank_team"] = by_unit["toi"].rank(ascending=False, method="min")
    out["mp_pp_toi_rank_team"] = by_unit["pp_toi"].rank(ascending=False, method="min")

    team_pp_seconds = role.groupby(["season_key", "team"], sort=False)["pp_ice"].transform("sum") / 5.0
    out["mp_pp_toi_share_of_team"] = _share(ice_pp, team_pp_seconds)

    # ── Production rates, all situations.
    out["mp_points_per60"] = _per60(a["I_F_points"], ice)
    out["mp_goals_per60"] = _per60(a["I_F_goals"], ice)
    out["mp_primary_assists_per60"] = _per60(a["I_F_primaryAssists"], ice)
    out["mp_secondary_assists_per60"] = _per60(a["I_F_secondaryAssists"], ice)
    out["mp_shots_per60"] = _per60(a["I_F_shotsOnGoal"], ice)
    out["mp_shot_attempts_per60"] = _per60(a["I_F_shotAttempts"], ice)
    out["mp_xgoals_per60"] = _per60(a["I_F_xGoals"], ice)
    out["mp_high_danger_shots_per60"] = _per60(a["I_F_highDangerShots"], ice)
    out["mp_high_danger_xgoals_per60"] = _per60(a["I_F_highDangerxGoals"], ice)
    out["mp_rebounds_per60"] = _per60(a["I_F_rebounds"], ice)
    out["mp_game_score_per_game"] = a["gameScore"] / gp

    # ── Shooting luck. The regression signal: a player finishing far above his
    #    expected goals got lucky and will come back; his shot volume will not.
    out["mp_shooting_pct"] = _share(a["I_F_goals"], a["I_F_shotsOnGoal"])
    out["mp_x_shooting_pct"] = _share(a["I_F_xGoals"], a["I_F_shotsOnGoal"])
    out["mp_goals_above_expected"] = a["I_F_goals"] - a["I_F_xGoals"]
    out["mp_goals_above_expected_per60"] = _per60(a["I_F_goals"] - a["I_F_xGoals"], ice)
    out["mp_high_danger_shot_share"] = _share(
        a["I_F_highDangerShots"],
        a["I_F_lowDangerShots"] + a["I_F_mediumDangerShots"] + a["I_F_highDangerShots"],
    )

    # ── Power play, rated against power-play time only. A fourth-liner who gets
    #    30 seconds of PP a night and a PP1 quarterback are finally comparable.
    out["mp_pp_points_per60"] = _per60(pp["I_F_points"], ice_pp)
    out["mp_pp_goals_per60"] = _per60(pp["I_F_goals"], ice_pp)
    out["mp_pp_shots_per60"] = _per60(pp["I_F_shotsOnGoal"], ice_pp)
    out["mp_pp_xgoals_per60"] = _per60(pp["I_F_xGoals"], ice_pp)
    out["mp_pp_onice_xgf_per60"] = _per60(pp["OnIce_F_xGoals"], ice_pp)
    out["mp_pp_point_share"] = _share(pp["I_F_points"], a["I_F_points"])

    # ── On-ice context: how much offence happens while this player is out
    #    there. The closest thing available to a linemate-quality feature.
    out["mp_onice_xg_pct"] = a["onIce_xGoalsPercentage"]
    out["mp_onice_xgf_per60"] = _per60(a["OnIce_F_xGoals"], ice)
    out["mp_onice_xga_per60"] = _per60(a["OnIce_A_xGoals"], ice)
    out["mp_ev_onice_xg_pct"] = ev["onIce_xGoalsPercentage"]
    out["mp_ev_onice_xgf_per60"] = _per60(ev["OnIce_F_xGoals"], ice_ev)
    out["mp_ev_points_per60"] = _per60(ev["I_F_points"], ice_ev)
    out["mp_ev_xgoals_per60"] = _per60(ev["I_F_xGoals"], ice_ev)
    out["mp_relative_xg_pct"] = a["onIce_xGoalsPercentage"] - a["offIce_xGoalsPercentage"]

    # ── Peripherals. The pool does not pay for these today, but they are role
    #    signals, and they are here the day the scoring rules change.
    out["mp_hits_per60"] = _per60(a["I_F_hits"], ice)
    out["mp_takeaways_per60"] = _per60(a["I_F_takeaways"], ice)
    out["mp_giveaways_per60"] = _per60(a["I_F_giveaways"], ice)
    out["mp_blocks_per60"] = _per60(a["shotsBlockedByPlayer"], ice)
    out["mp_pim_per60"] = _per60(a["penalityMinutes"], ice)
    out["mp_penalties_drawn_per60"] = _per60(a["penaltiesDrawn"], ice)
    out["mp_faceoffs_won_per60"] = _per60(a["I_F_faceOffsWon"], ice)

    return out.reset_index()


def goalie_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per goalie-season of `mp_g_*` columns.

    The headline number is **GSAx** -- goals saved above expected, the gap
    between the goals a goalie should have allowed given the shots he faced and
    the goals he actually allowed. It is the standard way to separate a goalie
    from the team defending in front of him, which raw save percentage cannot.

    Situations are read from the goalie's own team's point of view, so `4on5`
    is his team shorthanded, i.e. him facing the other side's power play.
    """
    if df.empty:
        return pd.DataFrame()

    sit = _by_situation(df)
    a = sit["all"]
    idx = a.index
    ev = sit.get("5on5", pd.DataFrame()).reindex(idx)
    sh = sit.get("4on5", pd.DataFrame()).reindex(idx)

    gp = a["games_played"].clip(lower=1)
    ice = a["icetime"]

    out = pd.DataFrame(index=idx)
    out["mp_team"] = a["team"]
    out["mp_g_games_played"] = a["games_played"]
    out["mp_g_toi_per_game"] = ice / 60 / gp

    # Workload and the quality of what gets through to him.
    out["mp_g_shots_faced_per60"] = _per60(a["ongoal"], ice)
    out["mp_g_xgoals_against_per60"] = _per60(a["xGoals"], ice)
    out["mp_g_goals_against_per60"] = _per60(a["goals"], ice)
    danger = a["lowDangerShots"] + a["mediumDangerShots"] + a["highDangerShots"]
    out["mp_g_high_danger_share"] = _share(a["highDangerShots"], danger)

    # GSAx, in three flavours: total (rewards a workhorse), per 60 and per shot
    # (both rate the goalie regardless of how much he played).
    saved_above_expected = a["xGoals"] - a["goals"]
    out["mp_g_gsax"] = saved_above_expected
    out["mp_g_gsax_per60"] = _per60(saved_above_expected, ice)
    out["mp_g_gsax_per_shot"] = _share(saved_above_expected, a["ongoal"])

    out["mp_g_save_pct"] = 1 - _share(a["goals"], a["ongoal"])
    out["mp_g_xsave_pct"] = 1 - _share(a["xGoals"], a["ongoal"])
    out["mp_g_rebound_rate"] = _share(a["rebounds"], a["ongoal"])
    out["mp_g_freeze_rate"] = _share(a["freeze"], a["ongoal"])

    # Even strength strips out special-teams noise; the shorthanded share is a
    # read on how often his team hands the opposition a power play.
    out["mp_g_ev_gsax_per60"] = _per60(ev["xGoals"] - ev["goals"], ev["icetime"])
    out["mp_g_ev_xgoals_against_per60"] = _per60(ev["xGoals"], ev["icetime"])
    out["mp_g_sh_toi_share"] = _share(sh["icetime"], ice)

    return out.reset_index()


def player_season_table(seasons: Iterable[str] | None = None,
                        refresh: bool = False) -> pd.DataFrame:
    """
    The join-ready table: **one row per (playerId, season)** with `mp_*`
    columns for skaters and `mp_g_*` columns for goalies.

    Skater and goalie columns never overlap, so each row carries one set and
    NaN for the other. Merge straight onto any player-season table:

        totals.merge(moneypuck.player_season_table(), on=["playerId", "season"],
                     how="left")

    These are the season's *own* numbers. Anything predicting that season has
    to lag them first -- `features.add_moneypuck` does exactly that.
    """
    seasons = list(seasons or HISTORY_SEASONS)
    skaters = skater_features(load_seasons("skaters", seasons, refresh=refresh))
    goalies = goalie_features(load_seasons("goalies", seasons, refresh=refresh))

    frames = [f for f in (skaters, goalies) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["playerId", "season"])

    out = pd.concat(frames, ignore_index=True)
    out["playerId"] = out["playerId"].astype("int64")
    out["season"] = out["season"].astype(str)
    return out.sort_values(["playerId", "season"]).reset_index(drop=True)


def feature_names(table: pd.DataFrame | None = None) -> list[str]:
    """The numeric `mp_*` columns -- everything this module offers a model."""
    table = player_season_table() if table is None else table
    return sorted(
        c for c in table.columns
        if c.startswith("mp_") and pd.api.types.is_numeric_dtype(table[c])
    )
