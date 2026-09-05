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
add/drops once games are being played, and **does not exist yet.**

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

## Layout

```
fantasy/
  config.py      seasons, paths, real season lengths
  scoring.py     the pool's scoring rules
  data.py        NHL API + per-season cache
  moneypuck.py   MoneyPuck advanced stats + per-season cache
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

- **The whole in-season model.** Nothing predicts `fantasy_remaining`. Half the
  project, and it needs to exist before opening night in October.
- **Season tracker tab** is a stub with a TODO list in it.
- **Goalies are half-modelled.** MoneyPuck adds GSAx, workload and shot
  quality faced (— `prev1_mp_g_*`), but there is still no team context and no
  starter share, which is most of what decides goalie wins.
- **Two-stage rate × games**, ranking objectives, floor/ceiling — described in
  `IDEAS.md`, none written.
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
