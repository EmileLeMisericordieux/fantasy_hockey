"""
Ways to predict a player's season.

Everything here matches one signature, so it drops straight into
`evaluate.walk_forward`:

    fit_predict(train_df, test_df, feature_cols, target) -> predictions

Whole frames rather than X and y, because a two-stage model wants to build its
own targets and a ranking model wants group sizes. Neither fits an (X, y) call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

DEFAULT_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    tree_method="hist",
)


# ─── Baselines ─────────────────────────────────────────────────────────────────
# Worth running before anything clever. Projection systems built on "last year,
# pulled toward the average" are hard to beat, and a model that cannot clear
# them is telling you something.


def make_baseline(kind: str = "weighted_average", **kw):
    def fit_predict(train, test, feature_cols, target):
        if kind == "last_season":
            # Whatever they did last year, they will do again.
            return test["prev1_fp_per82"].fillna(train[target].mean()).to_numpy()

        if kind == "weighted_average":
            # Weighted average of the last three seasons, pulled toward the
            # league average. `weight` is how much of the estimate is that
            # average -- tuning this one number goes a surprisingly long way.
            prior = kw.get("regress_to", float(train[target].mean()))
            w = kw.get("weight", 0.25)
            est = test["avg3_fp_per82"].fillna(prior).to_numpy()
            return (1 - w) * est + w * prior

        if kind == "positional_mean":
            # Every defenceman is the average defenceman. The floor.
            means = train.groupby("positionCode")[target].mean()
            return test["positionCode"].map(means).fillna(train[target].mean()).to_numpy()

        raise ValueError(f"unknown baseline: {kind}")

    return fit_predict


# ─── Gradient boosting ─────────────────────────────────────────────────────────


def make_gbm(**overrides):
    """A plain squared-error GBM on whatever features you pass it."""
    params = {**DEFAULT_PARAMS, **overrides}

    def fit_predict(train, test, feature_cols, target):
        model = XGBRegressor(**params)
        model.fit(train[feature_cols], train[target])
        return model.predict(test[feature_cols])

    return fit_predict


# ─── In-season ─────────────────────────────────────────────────────────────────
# Same signature, different problem: one row per player-game, predicting
# `fantasy_remaining` — the points a player has still to score this season.
# Used for add/drops, so what matters is the order of the players available
# today, not the size of the number.


def make_inseason_baseline(kind: str = "pace", **kw):
    """
    The arithmetic anyone would do by hand. Beat these or the model is noise.

      pace          What he has averaged this season, over the games left. The
                    projection every fantasy site shows, and it trusts twelve
                    games exactly as much as sixty.
      prior_season  Last season's per-game rate, over the games left. Ignores
                    this season entirely — the opposite mistake.
      blend         Shrink pace toward last season by sample size. Early on it
                    is mostly last season; by March it is mostly pace. `k` is
                    the number of games at which the two weigh the same.
    """
    def fit_predict(train, test, feature_cols, target):
        # Calendar games left, not 82 minus his own games played. See
        # features.add_cumulative -- the naive version quietly rewards the
        # injury-prone, which is the one thing these baselines must not do.
        remaining = test["team_games_left"].to_numpy(dtype=float)
        pace = test["fantasy_per_game_so_far"].fillna(0).to_numpy(dtype=float)

        if kind == "pace":
            return pace * remaining

        # Rookies have no prior season. Falling back to their pace is the
        # honest default: it is the only evidence that exists for them.
        prior = test["prev1_fp_per_game"].to_numpy(dtype=float)
        prior = np.where(np.isnan(prior), pace, prior)

        if kind == "prior_season":
            return prior * remaining

        if kind == "blend":
            k = float(kw.get("k", 20))
            g = test["games_so_far"].to_numpy(dtype=float)
            w = g / (g + k)
            return (w * pace + (1 - w) * prior) * remaining

        raise ValueError(f"unknown in-season baseline: {kind}")

    return fit_predict


def make_inseason_gbm(rate: bool = False, sample_every: int = 1, **overrides):
    """
    Gradient boosting on the in-season table.

    `rate=True` splits the problem in two: predict fantasy points *per game
    remaining*, then multiply by the games remaining. The direct model has to
    spend trees rediscovering that a player with sixty games left scores more
    than the same player with six, which is arithmetic, not skill. Handing it
    over lets every tree work on the part that is actually hard.

    `sample_every=n` trains on every nth game row. Consecutive rows for one
    player barely differ — the rolling windows move by one game — so this
    cuts the fit several-fold at little cost. Scoring always uses every row.
    """
    params = {**DEFAULT_PARAMS, **overrides}

    def fit_predict(train, test, feature_cols, target):
        if sample_every > 1:
            train = train.iloc[::sample_every]

        model = XGBRegressor(**params)
        if not rate:
            model.fit(train[feature_cols], train[target])
            return model.predict(test[feature_cols])

        gr_train = train["team_games_left"].clip(lower=1)
        model.fit(train[feature_cols], train[target] / gr_train)
        return model.predict(test[feature_cols]) * test["team_games_left"].clip(lower=0)

    return fit_predict


def feature_importance(train: pd.DataFrame, feature_cols: list[str],
                       target: str, top_n: int = 20, **overrides) -> pd.DataFrame:
    """What a model is leaning on."""
    model = XGBRegressor(**{**DEFAULT_PARAMS, **overrides})
    model.fit(train[feature_cols], train[target])
    return (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
