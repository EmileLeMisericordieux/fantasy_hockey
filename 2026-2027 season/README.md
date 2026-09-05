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
including goalies. Nothing needs the network unless `data/raw/` is deleted.

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
| position average only | 35.8 | 0.211 | 0.291 | 74.8% |
| repeat last season | 28.0 | 0.772 | 0.514 | 85.7% |
| weighted average, 3 seasons | 25.4 | 0.778 | **0.523** | 86.1% |
| untuned XGBoost | **21.4** | **0.821** | 0.506 | **86.7%** |

Two things fall out of this.

**Position alone gets you 74.8%.** Guess that every defenceman is the average
defenceman and you already collect three quarters of a perfect draft. So the
whole contest is the stretch from 74.8% to 100%, and everything above captures
less than half of it.

**XGBoost wins on RMSE and barely drafts better.** It beats a three-line
weighted average by 16% on error, and converts that into six tenths of a
percentage point of draft value — while still hitting *fewer* of the true top
fifty. Getting the numbers close and getting the order right are different jobs,
and the pool only pays for order. So move `pct_of_perfect@50` and `rank_corr`,
not RMSE.

Switching the target to `fantasy_points` flips it — XGBoost 85.3% against the
average's 76.2% — because a per-game average has no concept of who stays
healthy. Which target to use is a real decision, not a detail.

## Layout

```
fantasy/
  config.py      seasons, paths, real season lengths
  scoring.py     the pool's scoring rules
  data.py        NHL API + per-season cache
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

## Not built yet

- **The whole in-season model.** Nothing predicts `fantasy_remaining`. Half the
  project, and it needs to exist before opening night in October.
- **Season tracker tab** is a stub with a TODO list in it.
- **Goalies are in the data and nothing else.** No team context, no starter
  share. A full roster slot going unmodelled.
- **Two-stage rate × games**, ranking objectives, floor/ceiling — described in
  `IDEAS.md`, none written.
- **No rookies.** Every `prev*` feature is empty for a first-year player, so the
  `seasons_of_history >= 1` filter drops them.
- **Retirements.** The candidate list is everyone who played last season,
  including players now retired or in Europe.
- **Feature set is thin** — rolling averages, prior seasons, age, draft pick.
  Nothing on power play, ice time, linemates or schedule.

## Data quirks

- **2019-20 was 70 games and 2020-21 was 56.** Handled in
  `config.SEASON_LENGTH`; prefer `fp_per82` over raw totals.
- **A player can exceed 82 games.** Traded players inherit their new team's
  schedule — Tyson Barrie played 85 in 2022-23. Real, not a bug.
- **Join on `playerId`, never on name.** IDs are stable; spellings are not.
