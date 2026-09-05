"""
Ways to score a prediction.

Every evaluation reports these. In rough order of how much they matter here:

  pct_of_perfect@50  Draft this model's top 50 players. What share of the points
                     a perfect draft would have got do you end up with?
                     0.861 means 86.1%. THIS IS THE ONE THAT MATTERS.
  points_lost@50     The same thing in fantasy points instead of a percentage.
  rank_corr          Did we put players in the right order? 1.0 = perfect order,
                     0 = random. (Technically Spearman rank correlation.)
  top50_hit          Of our top 50 picks, what fraction were genuinely top 50?
  RMSE               Typical size of our error, in fantasy points. Lower is
                     better. Useful for debugging, but see the warning below.
  MAE                Same idea, less punishing about big misses.
  bias               Are we over- or under-predicting on average? + = too high.

**Careful with RMSE.** It measures whether your numbers land close. A draft does
not pay for that -- it pays for putting players in the right *order*. You need to
know Kucherov goes before Kreider; you never care by how much. A model can have
clearly better RMSE and still draft worse, and in this repo one already does.
When they disagree, believe pct_of_perfect@50.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ─── Metrics ───────────────────────────────────────────────────────────────────


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAE": float(np.abs(err).mean()),
        "bias": float(err.mean()),
    }


def rank_metrics(y_true, y_pred, ks=(25, 50, 100)) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {"rank_corr": float(spearmanr(y_true, y_pred).statistic)}

    order_pred = np.argsort(-y_pred)
    order_true = np.argsort(-y_true)
    for k in ks:
        if k > len(y_true):
            continue
        hit = len(set(order_pred[:k]) & set(order_true[:k])) / k
        out[f"top{k}_hit"] = float(hit)
    return out


def draft_regret(y_true, y_pred, n_picks: int = 50) -> dict:
    """
    Pretend we drafted the model's top `n_picks` players. Compare the points
    they actually scored against the best `n_picks` we could possibly have
    taken with hindsight.

    This is the closest thing to the real game, so it is the number to put in
    front of your friend.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if n_picks > len(y_true):
        n_picks = len(y_true)

    got = y_true[np.argsort(-y_pred)[:n_picks]].sum()
    best = np.sort(y_true)[::-1][:n_picks].sum()
    return {
        f"your_pts@{n_picks}": float(got),
        f"perfect_pts@{n_picks}": float(best),
        f"points_lost@{n_picks}": float(best - got),
        f"pct_of_perfect@{n_picks}": float(got / best) if best else float("nan"),
    }


def all_metrics(y_true, y_pred, n_picks: int = 50) -> dict:
    return {
        **regression_metrics(y_true, y_pred),
        **rank_metrics(y_true, y_pred),
        **draft_regret(y_true, y_pred, n_picks),
    }


# ─── Walk-forward validation ───────────────────────────────────────────────────


def walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    fit_predict,
    season_col: str = "season_start",
    min_train_seasons: int = 3,
    n_picks: int = 50,
) -> pd.DataFrame:
    """
    Test the way you would actually use the model: only ever predict a season
    using seasons that came before it.

        train on 2015-2018  -> predict 2019
        train on 2015-2019  -> predict 2020
        train on 2015-2020  -> predict 2021   ... and so on

    Returns one row per test season. Averaging them with `compare()` gives the
    single number for the leaderboard.

    This is why we do not use a normal random train/test split: a random split
    would let the model learn from 2024 to predict 2019, which you can never do
    in real life, and it would flatter every score.

    Plug in your own model by passing a function:

        def my_model(train, test, feature_cols, target):
            ...
            return predictions_for_test

    You get whole dataframes rather than just X and y, so you can use any column
    you like. The one rule: never read `target` out of `test`, that is the
    answer you are supposed to be predicting.
    """
    d = df.dropna(subset=[target]).copy()
    seasons = sorted(d[season_col].unique())
    rows = []

    for i, test_season in enumerate(seasons):
        if i < min_train_seasons:
            continue
        train = d[d[season_col] < test_season]
        test = d[d[season_col] == test_season]
        if train.empty or test.empty:
            continue

        pred = np.asarray(fit_predict(train, test, feature_cols, target), dtype=float)

        rows.append({
            "test_season": int(test_season),
            "n_train": len(train),
            "n_test": len(test),
            **all_metrics(test[target].to_numpy(), pred, n_picks),
        })

    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, name: str = "") -> pd.Series:
    """Average the per-season results into one comparable row."""
    num = results.select_dtypes("number").drop(columns=["test_season"], errors="ignore")
    s = num.mean()
    s.name = name
    return s


def compare(results_by_name: dict[str, pd.DataFrame],
            keep=("RMSE", "MAE", "rank_corr", "top50_hit", "pct_of_perfect@50")) -> pd.DataFrame:
    """Leaderboard across several approaches. This is the workshop scoreboard."""
    rows = [summarise(r, name) for name, r in results_by_name.items()]
    out = pd.DataFrame(rows)
    cols = [c for c in keep if c in out.columns]
    return out[cols].sort_values(cols[-1] if cols else out.columns[0], ascending=False)
