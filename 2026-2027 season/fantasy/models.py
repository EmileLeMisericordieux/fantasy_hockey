"""
Ways to predict a player's season.

Everything here matches one signature, so it drops straight into
`evaluate.walk_forward`:

    fit_predict(train_df, test_df, feature_cols, target) -> predictions

Whole frames rather than X and y, because a two-stage model wants to build its
own targets and a ranking model wants group sizes. Neither fits an (X, y) call.
"""

from __future__ import annotations

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
