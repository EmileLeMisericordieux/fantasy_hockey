# Backlog

Ranked roughly by payoff ÷ effort. Every item names where the data comes from,
because "we couldn't get the data" is how a day like this dies.

Claim an item by putting your name next to it. Register whatever you add in
`fantasy/features.py` so it gets picked up.

**MoneyPuck landed and moved several of these.** `fantasy/moneypuck.py` now
supplies expected goals, shot danger, on-ice rates and situational ice time for
all 11 seasons. Items 1, 2, 3 and 6 below are partly done as a result; what is
left of each is marked. See the MoneyPuck section of the README.

Unfamiliar terms are in the README glossary.

**The scoring rules bias everything toward defencemen.** A D goal is worth 3 and
an assist 2, versus 2 and 1 for forwards. A 50-point defenceman scores like a
75-point forward. Any ranking that ignores this drafts badly, and it makes
`positionCode` interactions unusually valuable.

---

## Do these first — new features

### 1. Power-play role  ✅ **largely done** — and it was the biggest win
Confirmed as predicted: this is now the largest MoneyPuck contribution to the
draft model. `prev1_mp_pp_toi_share_of_team` (the fraction of his team's power
play a player is on the ice for) and `prev1_mp_pp_toi_per_game` sit second and
third in feature importance behind raw prior scoring.
- Built: `mp_pp_toi_per_game`, `mp_pp_toi_share`, `mp_pp_toi_share_of_team`,
  `mp_pp_toi_rank_team`, `mp_pp_points_per60`, `mp_pp_xgoals_per60`.
- **Still open:** the *trend* within a season. Season summaries cannot show a
  player being promoted to PP1 in February, which is exactly the signal you
  want. Needs game-level data.

### 2. Ice time as opportunity  ✅ **largely done**
- Built: `mp_toi_per_game` split four ways (all / even strength / PP / PK), and
  `mp_toi_rank_team`, the rank among the player's own team's forwards or D —
  which is in the model's top five features.
- **Still open:** the within-season trend, and the breakout signal that comes
  from rising TOI on a young player. Both need game-level ice time.

### 3. Shooting-percentage regression  ✅ **the inputs exist**
MoneyPuck's expected-goals model is a much better regression signal than a
career shooting average, because it knows *where* the shots came from.
- Built: `mp_shooting_pct`, `mp_x_shooting_pct`, `mp_goals_above_expected`,
  `mp_xgoals_per60`, plus the shot-danger split. Sanity-checks correctly —
  Draisaitl +20.6 goals above expected in 2024-25, Hyman —11.0.
- **Still open:** the *use* of it. Nothing yet projects shot volume and
  finishing separately and multiplies them; the model just gets both columns.

### 4. Durability / games-played modelling
Season totals are `rate × games`, and nothing currently models the games half —
which systematically over-projects the injury-prone.
- `prev_games_pct` over the last three seasons, and its minimum
- Age interaction — availability falls off a cliff after ~34
- **Best version:** two-stage model, predict games played and per-game rate
  separately, then multiply. See "Two-stage" below.
- Source: game logs.

### 5. Age curve
Forwards peak around 24-27, defencemen a year or two later, and the decline is
not symmetric. `age` and `age_sq` exist but a tree model handles this awkwardly.
- Explicit `years_from_peak` by position
- Or fit the empirical curve from the data and use it as an offset
- Source: bios, already downloaded.

### 6. Goalies — half-modelled now
- Built: 18 `mp_g_*` columns, headlined by **GSAx** (goals saved above
  expected), which is the standard way to separate a goalie from the defence in
  front of him. Ranks Hellebuyck first in 2024-25, his Vezina and Hart season.
  Also workload, shot danger faced, and expected save percentage.
- **Still open, and it is the bigger half:** `starter_share` = games started ÷
  *team* games. Wins are mostly a team stat, and the backup/starter split is
  most of the variance. Needs team schedule data, which MoneyPuck's
  `teams.csv` or `client.stats.team_summary` would give you.

---

## Worth doing — a bit more work

### 7. Team offensive context
A player's output depends on the team around him, and it moves when he is traded
or when a team's whole forward group changes.
- Team goals-for per game, prior season
- `changed_team` flag, and the delta in team quality
- Source: `client.stats.team_summary`.

### 8. Opponent strength (in-season)
- Opponent goals-against per game, rolling
- Matters much more for weekly decisions than for season totals.
- Source: `client.stats.team_summary` joined on `opponentAbbrev`.

### 9. Schedule density (in-season)
Directly actionable for add/drops: a player with four games this week is worth
more than one with two.
- `games_next_7d`, `games_next_14d`, back-to-backs
- Source: `client.schedule.team_season_schedule` — verified available.

### 10. Consistency: floor versus ceiling
Two players averaging 1.0 fantasy points per game are not equivalent if one is
steady and the other alternates hat-tricks with healthy scratches.
- Standard deviation of per-game fantasy points, and P10/P90
- Matters for weekly head-to-head; matters less for season totals.
- Source: game logs.

### 11. Advanced tracking stats
`client.edge` exposes NHL EDGE tracking data — skating speed, distance covered.
Unexplored, possibly rich, possibly a dead end.
- Check how far back the coverage goes before committing time; EDGE is only a
  few seasons old.

---

## Changes to the model itself, rather than new features

### 1. Two-stage: rate × games  ⭐ likely the biggest single win
Model `fp_per_game` and `games_played` separately, multiply for the season total.
Fixes the injury blind spot and makes both halves individually debuggable. Use
it against the `fantasy_points` target — against `fp_per82` it has no advantage,
because that target has already divided games back out.

### 2. Optimise ranking, not error
The pool is won by ordering players correctly. XGBoost's `rank:pairwise` /
`rank:ndcg` optimise ordering directly. A model with worse RMSE can absolutely
draft better — `evaluate.py` reports `rank_corr` and `pct_of_perfect@50` so you can
see this happen.

### 3. Improve Edge (value over replacement)
Already in the dashboard in basic form: a player's projection minus what you
could still get at his position later. It is the real draft currency. Worth
refining — different pool sizes, bench slots, players eligible at two positions.

### 4. Floor and ceiling, not just one number
Instead of predicting one value, predict a *range*: a pessimistic case, a
middle, and an optimistic one. XGBoost does this with
`objective="reg:quantileerror"` and `quantile_alpha` set to 0.1, 0.5, 0.9.
Then draft the safe middle early and gamble on high ceilings late.
- Watch for the three models disagreeing — nothing forces the optimistic one to
  come out above the middle one, so sort them before displaying.

### 5. Simulate the season
Once you have ranges, run the season a few thousand times at random and answer
the question your friend actually has: *what is the probability this roster wins
the pool?* Much more compelling in a dashboard than a single number.

### 6. Trust small samples less
A player with 12 good games is much weaker evidence than a player with 70. Pull
short-sample players toward the average for their position, more aggressively
the fewer games they played. Cheap version: give the model `games_played` and
`seasons_of_history` and let it work this out itself.

---

## Known data quirks

- **2019-20 and 2020-21 were short** (70 and 56 games). `config.SEASON_LENGTH`
  handles this. Anything that assumes 82 will be badly wrong for those two
  seasons — prefer the `fp_per82` target over raw totals.
- **A player can exceed 82 games.** Traded players pick up their new team's
  schedule — Tyson Barrie played 85 in 2022-23. Not a bug; don't "fix" it.
- **Rookies have no prior season.** Every `prev*` feature is NaN. They are
  excluded by the `seasons_of_history >= 1` filter, which means *the model
  cannot draft rookies at all* — a real gap if the pool includes them.
- **Player IDs are stable; names are not.** Always join on `playerId`.
