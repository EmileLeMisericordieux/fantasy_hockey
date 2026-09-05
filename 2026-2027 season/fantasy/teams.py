"""
Team context, reconstructed from the game logs already on disk.

No new download. Every team fact here comes out of the player game logs:

  * **Who played whom, when.** `teamAbbrev` / `opponentAbbrev` / `gameId` on
    every row. Deduplicated, that is the season schedule.
  * **The score.** A goalie row carries `goalsAgainst`, so a team's goals
    against is its own goalies' total and its goals for is the opposition
    goalies' total. More reliable than summing skater goals, which misses
    shootout winners.
  * **The result.** A goalie row carries `decision` -- W, L, or O for an
    overtime loss. Two points for a win, one for an overtime loss.

From those three: standings as of any date, team strength, and what is left on
the schedule.

**Everything here is as of the morning of a game, never including it.** The
running totals are shifted by a day, matching the rule the in-season features
follow -- a row is what you knew before puck drop.

Knowing *who* a team still has to play is not hindsight: the NHL publishes the
full schedule in advance, so on any date the remaining opponents are genuinely
known. Knowing how those games *turn out* is hindsight, and nothing here uses
it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Two points for a win, one for losing in overtime or the shootout.
WIN_POINTS = 2.0
OTL_POINTS = 1.0


# ─── The team-game table ───────────────────────────────────────────────────────


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """
    One row per team-game: who, when, against whom, and how it finished.

    Needs goalie rows in `games` -- the score and the result are read off them.
    """
    goalies = games[games["positionCode"] == "G"]
    if goalies.empty:
        raise ValueError(
            "team_games needs goalie rows: the score and result are read from "
            "`goalsAgainst` and `decision`. Load with include_goalies=True."
        )

    dec = goalies["decision"].astype(str).str.upper()
    g = goalies.assign(
        _ga=pd.to_numeric(goalies["goalsAgainst"], errors="coerce").fillna(0),
        _win=(dec == "W").astype(float),
        _otl=(dec == "O").astype(float),
    )

    # A team can dress two goalies in one game, so sum over the team's goalies.
    tg = (
        g.groupby(["season", "gameId", "teamAbbrev"], as_index=False)
        .agg(gameDate=("gameDate", "first"),
             opponent=("opponentAbbrev", "first"),
             home=("homeRoadFlag", "first"),
             ga=("_ga", "sum"),
             win=("_win", "max"),
             otl=("_otl", "max"))
        .rename(columns={"teamAbbrev": "team"})
    )

    # Goals for = what the other team's goalies let in.
    opp = tg[["season", "gameId", "team", "ga"]].rename(
        columns={"team": "opponent", "ga": "gf"}
    )
    tg = tg.merge(opp, on=["season", "gameId", "opponent"], how="left")
    tg["gf"] = tg["gf"].fillna(0)

    tg["points"] = tg["win"] * WIN_POINTS + tg["otl"] * OTL_POINTS
    tg["is_home"] = (tg["home"].astype(str).str.upper() == "H").astype(int)
    tg["gameDate"] = pd.to_datetime(tg["gameDate"])
    return tg.sort_values(["season", "team", "gameDate"]).reset_index(drop=True)


# ─── Standings as of a date ────────────────────────────────────────────────────


def daily_standings(tg: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (season, date, team): where that team stood on the **morning**
    of that date, before any of the day's games.

    Built on a daily grid rather than per game so that every team can be
    compared on the same date even when they have played different numbers of
    games.

    Columns:
      team_games_played   games so far
      team_points         standings points so far
      team_points_pct     points as a share of the maximum available; the fair
                          comparison mid-season, when games in hand are common
      team_gf_per_game / team_ga_per_game
      team_goal_diff_per_game   goals for minus against, per game. **This is
                          the "team performance" number**: higher is better, so
                          "below the league mean" reads as the weaker half.
      team_rank           1 = best, by points, then goal difference
    """
    rows = []
    for season, g in tg.groupby("season", sort=False):
        dates = pd.date_range(g["gameDate"].min(), g["gameDate"].max(), freq="D")
        wide = g.pivot_table(
            index="gameDate", columns="team",
            values=["points", "gf", "ga"], aggfunc="sum",
        )
        played = g.pivot_table(index="gameDate", columns="team",
                               values="gameId", aggfunc="count")
        wide = wide.reindex(dates).fillna(0.0)
        played = played.reindex(dates).fillna(0.0)

        # Shift by a day so a date's standing excludes that date's games.
        cum = wide.cumsum().shift(1).fillna(0.0)
        cum_played = played.cumsum().shift(1).fillna(0.0)

        long = pd.DataFrame({
            "team_points": cum["points"].stack(),
            "team_gf": cum["gf"].stack(),
            "team_ga": cum["ga"].stack(),
            "team_games_played": cum_played.stack(),
        })
        long.index.names = ["gameDate", "team"]
        long = long.reset_index()
        long["season"] = season
        rows.append(long)

    out = pd.concat(rows, ignore_index=True)

    gp = out["team_games_played"].clip(lower=1)
    out["team_gf_per_game"] = out["team_gf"] / gp
    out["team_ga_per_game"] = out["team_ga"] / gp
    out["team_goal_diff_per_game"] = out["team_gf_per_game"] - out["team_ga_per_game"]
    out["team_points_pct"] = out["team_points"] / (2 * gp)

    # Nobody has played yet on opening day; leave those undefined rather than
    # calling all 32 teams exactly average.
    unplayed = out["team_games_played"] == 0
    for c in ("team_gf_per_game", "team_ga_per_game",
              "team_goal_diff_per_game", "team_points_pct"):
        out.loc[unplayed, c] = np.nan

    out["team_rank"] = (
        out.groupby(["season", "gameDate"])["team_points"]
        .rank(ascending=False, method="min")
    )
    out.loc[unplayed, "team_rank"] = np.nan

    # The bar that decides "poor performing" for the schedule features below.
    out["league_mean_goal_diff"] = (
        out.groupby(["season", "gameDate"])["team_goal_diff_per_game"]
        .transform("mean")
    )
    return out


# ─── What is left on the schedule ──────────────────────────────────────────────


def remaining_schedule(tg: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    """
    For every team-game, what the rest of the season looks like from that
    morning onward -- **including** the game about to be played, matching how
    `fantasy_remaining` counts.

    Opponent quality is measured as of the decision date, not with hindsight:
    for a game on 1 December, every remaining opponent is judged by where it
    stood on 1 December.

      team_games_left            games from this one to the end of the season
      opp_ga_per_game_left       mean goals conceded by the remaining opponents.
                                 High means a soft run of fixtures.
      opp_goal_diff_left         mean strength of the remaining opponents.
                                 Low means an easy run.
      games_left_vs_weak         how many of those games are against teams below
                                 the league mean on goal difference
      pct_left_vs_weak           the same as a share of the games left
    """
    # Expand each team's season into (this game, every game from here on).
    pairs = []
    for (season, team), g in tg.groupby(["season", "team"], sort=False):
        dates = g["gameDate"].to_numpy()
        opps = g["opponent"].to_numpy()
        n = len(dates)
        i, j = np.triu_indices(n)          # j >= i: this game onward
        pairs.append(pd.DataFrame({
            "season": season,
            "team": team,
            "gameDate": dates[i],
            "opponent_left": opps[j],
        }))
    pairs = pd.concat(pairs, ignore_index=True)

    # Judge each remaining opponent as of the decision date.
    look = standings[["season", "gameDate", "team",
                      "team_ga_per_game", "team_goal_diff_per_game",
                      "league_mean_goal_diff"]].rename(
        columns={"team": "opponent_left",
                 "team_ga_per_game": "_opp_ga",
                 "team_goal_diff_per_game": "_opp_diff"}
    )
    pairs = pairs.merge(look, on=["season", "gameDate", "opponent_left"], how="left")
    pairs["_weak"] = (pairs["_opp_diff"] < pairs["league_mean_goal_diff"]).astype(float)
    pairs.loc[pairs["_opp_diff"].isna(), "_weak"] = np.nan

    out = pairs.groupby(["season", "team", "gameDate"], as_index=False).agg(
        team_games_left=("opponent_left", "size"),
        opp_ga_per_game_left=("_opp_ga", "mean"),
        opp_goal_diff_left=("_opp_diff", "mean"),
        games_left_vs_weak=("_weak", "sum"),
    )
    out["pct_left_vs_weak"] = out["games_left_vs_weak"] / out["team_games_left"].clip(lower=1)
    return out


def team_context(games: pd.DataFrame) -> pd.DataFrame:
    """
    Everything above, joined into one row per (season, team, gameDate).

    Merge onto player game rows with those three keys.
    """
    tg = team_games(games)
    standings = daily_standings(tg)
    left = remaining_schedule(tg, standings)

    ctx = tg[["season", "team", "gameDate"]].drop_duplicates()
    ctx = ctx.merge(standings, on=["season", "team", "gameDate"], how="left")
    ctx = ctx.merge(left, on=["season", "team", "gameDate"], how="left")
    return ctx.drop(columns=["team_gf", "team_ga"], errors="ignore")
