# Fantasy Hockey — 2026-27

Predict which NHL players are worth drafting, then keep the roster sharp through
the season. The data is downloaded and there is a working way to measure whether
an idea helped, so the day can go into features instead of plumbing.

## Setup

```bash
cd "2026-2027 season"
python -m pip install -r requirements.txt
```

Data is already on disk — 11 seasons, 2015-16 through 2025-26, 515k player-games
including goalies, plus MoneyPuck's advanced stats for the same 11 seasons.
Nothing needs the network unless `data/raw/` is deleted.

If it is, two scripts refill it:

```bash
python scripts/fetch_data.py        # NHL game logs and bios (slow, ~1h)
python scripts/fetch_moneypuck.py   # MoneyPuck advanced stats (~15s)
```

> **Always launch with `python -m streamlit`, not bare `streamlit`.**
> There can be several Python installs on one machine, each with its own
> packages. Bare `streamlit` runs whichever one PATH finds first, which is often
> not the one `python` points at — you get `ModuleNotFoundError: No module named
> 'plotly'` even though plotly is installed. `python -m` forces the interpreter
> `python` resolves to. Same reason for `python -m pip` above.
>
> Check which one you are on: `python -c "import sys; print(sys.executable)"`

## Glossary

Read this once and the rest of the repo makes sense.

| Term | What it means |
|---|---|
| **fantasy points** | The pool's currency. Forward: goal 2, assist 1. Defence: goal 3, assist 2. Hat trick +3. Goalie win 3. Set in `scoring.py` — **confirm these against the real pool rules.** |
| **fp** | Short for fantasy points. |
| **`fp_per82`** | Fantasy points scaled to a full 82-game season. Measures how good a player is *per game*, so an injured player isn't punished twice and the short COVID seasons stay comparable. |
| **`fantasy_points`** | The raw season total — what the pool actually pays. Rewards players who stay healthy. |
| **Edge** | How many points better a player is than **the worst starter at his position**. In a 10-team pool with 2 centres each, only 20 centres get drafted; beating the 20th is what earns you anything. **Draft by Edge, not by projection.** (Called VORP elsewhere.) |
| **`pct_of_perfect@50`** | Draft this model's top 50. What share of a *perfect* draft's points do you end up with? **The number that matters.** |
| **`rank_corr`** | Did we put players in the right order? 1.0 = perfect, 0 = random. |
| **RMSE** | Typical size of our error, in fantasy points. Lower is better — but it is **not** the goal, see "The bar" below. |
| **walk-forward** | Testing honestly: train only on seasons *before* the one you predict, then step forward a year and repeat. A random train/test split would let the model learn from 2024 to predict 2019 and flatter every score. |
| **weighted average** | Baseline model: blend the last 3 seasons, recent years counting more. Three lines of code, and hard to beat. |
| **GBM** | Gradient boosting (XGBoost). Builds many small decision trees, each fixing the previous one's mistakes. |
| **before_season / during_season** | A tag on every feature saying when you'd know it. The draft model only gets `before_season` — on draft day, nothing from the coming season has happened yet. |
| **TOI** | Time on ice, i.e. minutes played. The best single clue to how much a coach trusts a player. |
| **PP / PP1** | Power play; PP1 is the top power-play unit. Being on it is worth a lot of fantasy points. |
| **per 60** | A rate stat: production scaled to 60 minutes of ice time, so a fourth-liner and a star are compared fairly. |
| **xG** | Expected goals. What a shot was *worth* given where and how it was taken. Sums to "how many goals should this player have scored". |
| **GSAx** | Goals saved above expected. A goalie's xG faced minus goals allowed. Separates the goalie from the defence in front of him, which save percentage cannot. |
| **PP share of team** | The fraction of his team's power-play time a player is on the ice for. PP1 regulars sit near 0.7, PP2 near 0.3. The cleanest read on power-play role there is. |

## Start here

**1. Check the scoring rules** in `fantasy/scoring.py` — currently forwards 2/1,
defence 3/2, hat trick +3, goalie win 3. Confirm them against the actual pool.
Every number in this repo depends on them.

**2. See where the bar is.**

```bash
python -c "
from fantasy import datasets, evaluate, models
df = datasets.draft_table()
df = df[(df.seasons_of_history>=1) & (df.games_played>=20)]
feats = datasets.feature_columns(df, task='draft')
res = {k: evaluate.walk_forward(df, feats, 'fp_per82', f) for k, f in {
    'weighted_average': models.make_baseline('weighted_average'),
    'gbm':              models.make_gbm(),
}.items()}
print(evaluate.compare(res).round(3))
"
```

**3. Open the dashboard** and look at the draft board it produces today.

```bash
python -m streamlit run app.py
```

**4. Pick something from `IDEAS.md`** and build it. That's the backlog, roughly
ranked, with the data source named for each item.

## Two problems, not one

| | Draft model | In-season model |
|---|---|---|
| One row per | player-season | player-game |
| Can use | prior seasons only | everything played so far |
| Predicts | `fp_per82` or `fantasy_points` | `fantasy_remaining` |
| Built by | `datasets.draft_table()` | `datasets.inseason_table()` |
| Used | October, once | continuously |

The draft model matters most — it decides the roster. The in-season model drives
add/drops once games are being played; a first version now exists, see
"The bar, in-season" below.

## The bar

Walk-forward across 2019-2025, predicting `fp_per82`:

| | RMSE | rank_corr | top-50 hit | % of perfect draft |
|---|---|---|---|---|
| position average only | 36.3 | 0.225 | 0.423 | 79.6% |
| repeat last season | 29.2 | 0.776 | 0.534 | 85.9% |
| weighted average, 3 seasons | 26.4 | 0.783 | 0.546 | 86.3% |
| untuned XGBoost | 22.2 | 0.827 | 0.509 | 85.9% |
| XGBoost + MoneyPuck | **21.9** | **0.832** | **0.551** | **87.8%** |

All five re-measured in one pass, so they are comparable with each other.

Three things fall out of this.

**Position alone gets you 79.6%.** Guess that every defenceman is the average
defenceman and you already collect four fifths of a perfect draft. So the whole
contest is the stretch from 79.6% to 100%, and everything above captures less
than half of it.

**RMSE and draft value are different jobs.** Plain XGBoost beats a three-line
weighted average by 16% on error and still drafts *worse* — 85.9% against
86.3%, hitting fewer of the true top fifty. Getting the numbers close and
getting the order right are not the same thing, and the pool only pays for
order. So move `pct_of_perfect@50` and `rank_corr`, not RMSE.

**MoneyPuck is what breaks the tie.** Adding it is the first change that clears
the weighted average on *both* — 87.8% of a perfect draft and 0.551 of the
true top fifty. Roughly half the model's feature importance now sits in
MoneyPuck columns, and the biggest of them is power-play deployment: how much
of his team's power play a player is actually on the ice for. See "MoneyPuck"
below.

Switching the target to `fantasy_points` moves everything down but keeps the
order — MoneyPuck 85.2%, plain XGBoost 84.2%, the weighted average 73.1% —
because a per-game average has no concept of who stays healthy. Which target to
use is a real decision, not a detail.

## The bar, in-season

Different question, different scoreboard. Predicting `fantasy_remaining`, walk
forward by season, then score at three points *inside* each test season — 25%,
50% and 75% of the way through. Eight test seasons, ~800 players in the field
each time.

| | RMSE | rank_corr | top-25 hit | % of perfect @30 |
|---|---|---|---|---|
| blend (pace shrunk toward last season) | 16.5 | 0.724 | 0.340 | 69.3% |
| GBM, before the team/usage features | 11.4 | 0.779 | 0.417 | 80.3% |
| GBM | 11.1 | 0.784 | **0.430** | 81.3% |
| GBM, rate x games | **11.0** | **0.790** | 0.423 | **81.4%** |

```bash
python -c "
from fantasy import datasets, evaluate, models
df = datasets.inseason_table()
feats = datasets.feature_columns(df, task='inseason')
res = {k: evaluate.walk_forward_inseason(df, feats, 'fantasy_remaining', f) for k, f in {
    'blend': models.make_inseason_baseline('blend'),
    'gbm':   models.make_inseason_gbm(sample_every=10),
}.items()}
print(evaluate.compare(res, keep=('RMSE','rank_corr','top25_hit','pct_of_perfect@30')).round(3))
"
```

**Never score this by pooling every player-game.** `fantasy_remaining` falls
toward zero as a season runs out, so a model that knows nothing but the date
ranks the pooled rows almost perfectly. That number measures the calendar, not
the player. You never choose between a player in October and a player in March
— you choose among the players in front of you *today*, which is what the
checkpoints reproduce.

**The gap is real but it is smaller than it first looks.** The first version of
`pace` divided 82 by the player's *own* games played, which quietly hands extra
nights to exactly the players least likely to be there for them, and made the
GBM look 26 points better. Counting the calendar instead (`team_games_left`)
took `pace` from 44.5% to 66.1%. If a baseline looks that bad, suspect the
baseline.

**Where the model earns it is October, not April.** Broken out by point in the
season, `pct_of_perfect@30`:

| | 25% in | 50% in | 75% in |
|---|---|---|---|
| blend | 68.4% | 71.4% | 68.0% |
| GBM, before the team/usage features | 84.8% | 81.0% | 74.9% |
| GBM | **84.6%** | **83.0%** | **76.3%** |

Early on, pace is a dozen games of noise and the only real evidence about a
player is last season — which is what the GBM has and the baselines mostly do
not. By March everyone's pace has stabilised and the edge halves. Note this is
the reverse of where you might expect: the model is most valuable when you know
least.

**Rate x games is a coin flip so far.** Predicting points per game left and
multiplying, rather than predicting the total directly, wins by a tenth of a
point. It is better late and worse early. Worth keeping because it is the
honest shape of the problem, not because it has paid yet.

## What the in-season model leans on

The second wave of features — power-play usage, team standings, the remaining
schedule, availability and expected goals — moved `pct_of_perfect@30` from
80.3% to 81.3%. Modest overall, and **all of it arrives after Christmas**:
+2.0 points at the halfway mark, +1.4 in the run-in, and nothing at all in
October, where the model already ran on last season. That is what you would
expect from features that need games played before they mean anything.

The importance is concentrated to a degree worth knowing about. Of 31 new
columns, five do essentially all the work:

| Rank of 320 | Feature | What it is |
|---|---|---|
| **1** | `expected_goals_left` | rolling expected goals x expected games left |
| **4** | `expected_games_left` | schedule games left, scaled by how available he has been |
| **8** | `team_games_left` | exact games left, off the real schedule |
| **11** | `roll20_pp_toi` | power-play minutes a game, last 20 |
| **16** | `roll5_pp_toi` | power-play minutes a game, last 5 |

That is the two-stage shape — **a rate times a count of games** — arriving on
its own. The model was previously spending trees rediscovering it.

The other twenty-six are close to unused, and three of them are worth
explaining rather than deleting:

- **`pp_toi_delta5` (rank 304).** The *change* in power-play time adds almost
  nothing once the model has the *level*. That is not a bug in the feature: a
  player promoted to PP1 in November simply has high `roll20_pp_toi` by
  December, and the promotion itself only carries information for a few weeks.
  It stays because it is the right feature for a *weekly* question, which this
  model is not answering.
- **Opponent quality and schedule strength (ranks 163-308).** `fantasy_remaining`
  spans about forty games, and over forty games the schedule averages out to
  roughly the same for everyone. Strength of schedule is a real effect over
  *one week*; it is nearly nothing over half a season.
- **`changed_team` (rank 316, importance 0.0000).** It fires on 3.4% of rows,
  and `team_rank` (286) is collinear with the goal-difference columns that beat
  it. Rare and duplicated is a bad combination.

None of this makes them wrong to have built. It makes them the wrong features
for *this* target, and the right ones to reach for when the add/drop advisor
starts answering "who helps me most **this week**".

## Where team context comes from

`fantasy/teams.py`, and it needs **no new download**. Everything is
reconstructed from the player game logs already on disk:

- **Who played whom** — `teamAbbrev` / `opponentAbbrev` / `gameId`, deduplicated,
  is the schedule.
- **The score** — a goalie row carries `goalsAgainst`, so a team's goals against
  is its own goalies' total and its goals for is the opposition goalies'.
- **The result** — a goalie row carries `decision`, W / L / O.

Checked against the real 2024-25 table: all 32 teams, 82 games each, Winnipeg
first on 116 points, San Jose last on 52, and total goals for equal to total
goals against. Every figure is as of the *morning* of a game, never including
it. Knowing the remaining opponents is not hindsight — the NHL publishes the
schedule in advance — but knowing how those games turn out would be, and
nothing uses it.

Per-game power-play ice time and per-game expected goals are the one genuinely
new download: `moneypuck.load_game_logs()`, about 13MB for eleven seasons.
Unlike the season summaries these are **stored trimmed, not raw** — the full
files are a quarter of a gigabyte. Widen `_GAME_KEEP` and re-fetch if you need
more columns.

## Layout

```
fantasy/
  config.py      seasons, paths, real season lengths
  scoring.py     the pool's scoring rules
  data.py        NHL API + per-season cache
  moneypuck.py   MoneyPuck advanced stats, season and game level
  teams.py       standings, team strength and schedule, from the game logs
  features.py    feature builders — the thin part, and the work
  datasets.py    assembles the draft table and the in-season table
  models.py      baselines and a plain GBM, all one signature
  evaluate.py    walk-forward testing and the metrics
app.py           dashboard: draft board, model lab, season tracker
IDEAS.md         the backlog
```

## Adding a feature

1. Write the builder in `fantasy/features.py`.
2. Tag it: `register("my_column", "before_season")` — or `"during_season"` if it
   needs games already played. Untagged columns are ignored entirely.
3. `rm data/processed/*` to rebuild the tables.
4. Rerun the leaderboard and see whether it moved.

## MoneyPuck

`fantasy/moneypuck.py` pulls [moneypuck.com](https://moneypuck.com/data.htm),
which publishes what the NHL API does not: an expected-goals model, shot-danger
buckets, on-ice rates, and ice time **split by situation** — so power-play
deployment stops being a guess.

Two things make it cheap to use. **MoneyPuck's `playerId` is the NHL player
id**, so everything joins on `playerId` + `season` with no name matching. And
there is **one row per player-season, not per team** — traded players are
already consolidated, credited to the team they finished with.

Coverage is 11 of 11 seasons and 11,196 of 11,200 player-seasons. The four
misses are 1-2 game cameos worth 0-3 fantasy points, dropped by the
`games_played >= 20` filter anyway.

```bash
python scripts/fetch_moneypuck.py            # ~15s, all 11 seasons
python scripts/fetch_moneypuck.py --refresh  # re-download; use in-season
```

`data/raw/moneypuck_<kind>_<season>.parquet` holds the CSV as published — all
154 skater columns, all five situations — so curating a new feature never
means re-downloading. `moneypuck.player_season_table()` collapses that to one
join-ready row per player-season of 62 `mp_*` columns, all of them rates,
shares or per-game figures.

**Nothing raw is a feature.** Those columns are the season's own numbers, and
handing the draft model those would be handing it the answer.
`features.add_prior_season_features` lags every one of them into `prev1_mp_*`,
`prev2_mp_*`, `prev3_mp_*`, and *those* are what get registered — 199 of the
draft model's 244 features.

To A/B the whole block, filter on the name:

```python
feats = datasets.feature_columns(df, task="draft")
without_mp = [c for c in feats if "mp_" not in c]
```

What the model actually leans on, in importance order: `mp_game_score_per_game`,
`mp_pp_toi_share_of_team`, `mp_pp_toi_per_game`, `mp_toi_rank_team`. Power-play
role, three different ways, exactly as `IDEAS.md` predicted.

Still on the table and not fetched: `lines.csv` (line combinations, the real
linemate feature) and `teams.csv` (team offensive context, `IDEAS.md` #7). Both
are one URL each in `moneypuck.BASE_URL`.

## Not built yet

- **The in-season model is a first version only.** It trains and scores
  honestly, but nothing consumes it: no add/drop screen, no "who should I pick
  up this week", no weekly schedule weighting. And it has never seen a live
  season — 2026-27 has not started, so it has only ever been tested on
  history.
- **Season tracker tab** now replays a finished season on a date slider, so
  the model can be inspected before opening night. What it still lacks is an
  **add/drop advisor** — nothing turns the projection into "pick up this
  player".
- **Injuries are invisible.** Nothing knows a player is hurt; the model infers
  it from ice time drying up, several games late. The evaluation is kinder
  still — a player who never returns after a checkpoint drops out of the
  field entirely instead of scoring the zero he really earned.
- **Goalies are half-modelled.** MoneyPuck adds GSAx, workload and shot
  quality faced (— `prev1_mp_g_*`), but there is still no team context and no
  starter share, which is most of what decides goalie wins.
- **Ranking objectives and floor/ceiling** — described in `IDEAS.md`, neither
  written. Two-stage rate × games now exists for the in-season model only
  (`models.make_inseason_gbm(rate=True)`); the draft model still predicts
  season totals directly.
- **No rookies.** Every `prev*` feature is empty for a first-year player, so the
  `seasons_of_history >= 1` filter drops them.
- **Retirements.** The candidate list is everyone who played last season,
  including players now retired or in Europe.
- **Still no schedule or real linemates.** MoneyPuck covers power play, ice
  time, shot quality and on-ice context; `lines.csv` and the team schedule are
  the two obvious gaps left.

## Data quirks

- **2019-20 was 70 games and 2020-21 was 56.** Handled in
  `config.SEASON_LENGTH`; prefer `fp_per82` over raw totals.
- **A player can exceed 82 games.** Traded players inherit their new team's
  schedule — Tyson Barrie played 85 in 2022-23. Real, not a bug.
- **Join on `playerId`, never on name.** IDs are stable; spellings are not.
