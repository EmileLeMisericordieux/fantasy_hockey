"""
Inspection script for the game-level fantasy hockey pipeline.

Builds a small real dataset (3 seasons), trains a model, and prints a
detailed report at every stage: dataset, features, model, predictions.

Usage:
    python test_inspection.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from feature_engineering import FantasyDataEngineer

# ── Config ─────────────────────────────────────────────────────────────────────

TRAIN_SEASONS     = ["20212022", "20222023", "20232024"]
TRAIN_CUTOFF_YEAR = 2022   # train on seasons starting ≤ this year
TEST_SEASON_YEAR  = 2023   # test on games from this season start year

MIN_GAMES = 5

RAW_STAT_COLS = [
    "goals", "assists", "points", "shots", "toi", "timeOnIce", "toi_minutes",
    "powerPlayGoals", "powerPlayAssists", "shortHandedGoals", "shortHandedAssists",
    "plusMinus", "pim", "decision", "homeRoadFlag", "fantasy_points", "p60",
]
META_COLS = [
    "playerId", "seasonId", "gameId", "season", "season_start_year", "gameDate",
    "skaterFullName", "lastName", "teamAbbrevs", "opponentTeamAbbrev",
]
TARGET    = "season_fantasy_remaining"
TOTAL_COL = "season_fantasy_total"
CAT_COLS_CANDIDATES = ["positionCode", "shootsCatches"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def header(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def warn(msg: str):
    print(f"  ⚠  WARNING: {msg}")

def ok(msg: str):
    print(f"  ✓  {msg}")


# ── Build data & model ─────────────────────────────────────────────────────────

def build_game_df(use_cache: bool = True) -> pd.DataFrame:
    cache_key = "_".join(TRAIN_SEASONS)
    cache_path = Path(f"cache_game_df_{cache_key}.parquet")

    if use_cache and cache_path.exists():
        print(f"Loading cached dataset from {cache_path}...")
        df = pd.read_parquet(cache_path)
        if TARGET in df.columns:
            return df
        print(f"  Cache is stale (missing '{TARGET}'), rebuilding...")

    engineer = FantasyDataEngineer(seasons=TRAIN_SEASONS, limit=100, debug=False)
    df = engineer.build_game_dataset()
    df.to_parquet(cache_path, index=False)
    print(f"Dataset cached to {cache_path}")
    return df


def build_model_bundle(game_df: pd.DataFrame) -> dict:
    df = game_df[game_df["games_in_season"] >= MIN_GAMES].copy()

    cat_cols = [c for c in CAT_COLS_CANDIDATES if c in df.columns]
    df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    drop_cols   = set(RAW_STAT_COLS + META_COLS + [TARGET, TOTAL_COL]) & set(df_model.columns)
    feature_cols = [
        c for c in df_model.columns
        if c not in drop_cols and df_model[c].dtype != object
    ]

    train = df_model[df_model["season_start_year"] <= TRAIN_CUTOFF_YEAR].dropna(subset=[TARGET])
    test  = df_model[df_model["season_start_year"] == TEST_SEASON_YEAR].dropna(subset=[TARGET])

    X_train, y_train = train[feature_cols], train[TARGET]
    X_test,  y_test  = test[feature_cols],  test[TARGET]
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:quantileerror",
        quantile_alpha=0.65,
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    # reconstruct total: cum_fantasy_so_far + predicted_remaining
    pred_remaining = model.predict(X_test).clip(min=0)
    y_pred = test["cum_fantasy_so_far"].values + pred_remaining
    y_actual_total = test[TOTAL_COL].values if TOTAL_COL in test.columns else y_test.values + test["cum_fantasy_so_far"].values

    test_meta = test[["season_start_year"]].copy()
    if "skaterFullName" in df_model.columns:
        test_meta["skaterFullName"] = test["skaterFullName"]
    test_meta["y_actual"] = y_actual_total
    test_meta["y_pred"]   = y_pred
    test_meta["error"]    = y_pred - y_test.values

    return {
        "X_train":      X_train,
        "y_train":      y_train,
        "X_test":       X_test,
        "y_test":       y_test,
        "model":        model,
        "y_pred":       y_pred,
        "feature_cols": feature_cols,
        "test_meta":    test_meta,
    }


# ── Inspection sections ────────────────────────────────────────────────────────

def inspect_dataset(game_df: pd.DataFrame):
    header("DATASET")

    n_players = game_df["playerId"].nunique()
    n_seasons = game_df["season"].nunique()
    n_games   = game_df["gameId"].nunique() if "gameId" in game_df.columns else "N/A"
    print(f"  Shape          : {game_df.shape}")
    print(f"  Unique players : {n_players}")
    print(f"  Unique seasons : {n_seasons}")
    print(f"  Unique games   : {n_games}")

    print("\n  Columns (name | dtype | missing count | missing %):")
    for col in sorted(game_df.columns):
        n_null = game_df[col].isna().sum()
        pct    = 100 * n_null / len(game_df)
        print(f"    {col:<42} {str(game_df[col].dtype):<12}  {n_null:>6} ({pct:.1f}%)")

    print(f"\n  Target ({TARGET}) distribution:")
    pts = game_df[TARGET].dropna()
    print(pts.describe().to_string())
    print(f"\n  Value counts (top 15):")
    print(pts.value_counts().head(15).to_string())

    if pts.min() < 0:
        warn("Negative fantasy_points found — check scoring logic")
    else:
        ok("All fantasy_points >= 0")

    print("\n  5 random game rows:")
    show = [c for c in ["skaterFullName", "season", "gameDate", "positionCode", "goals", "assists", TARGET] if c in game_df.columns]
    print(game_df[show].dropna(subset=[TARGET]).sample(5, random_state=42).to_string(index=False))

    if "positionCode" in game_df.columns:
        print("\n  Fantasy points by position:")
        breakdown = (
            game_df.groupby("positionCode")[TARGET]
            .agg(count="count", mean="mean", std="std", median="median")
        )
        print(breakdown.to_string())

    gps = game_df.groupby(["playerId", "season"]).size()
    print(f"\n  Games per player-season — mean: {gps.mean():.1f}, min: {gps.min()}, max: {gps.max()}")


def inspect_features(game_df: pd.DataFrame):
    header("FEATURES")

    roll_cols = [c for c in game_df.columns if c.startswith("roll")]
    print(f"  Rolling feature columns ({len(roll_cols)}):")
    for c in roll_cols:
        print(f"    {c}")

    if roll_cols:
        print("\n  Rolling feature descriptive stats:")
        print(game_df[roll_cols].describe().T[["mean", "std", "min", "max"]].to_string())

    if "roll5_fantasy_points" in game_df.columns and "fantasy_points" in game_df.columns:
        multi = game_df[game_df.groupby(["playerId", "season"])["playerId"].transform("count") > 5]
        match_rate = (multi["roll5_fantasy_points"] == multi["fantasy_points"]).mean()
        print(f"\n  Leakage check — fraction where roll5_fantasy_points == fantasy_points: {match_rate:.3f}")
        if match_rate >= 0.5:
            warn("High match rate — shift(1) may not be applied correctly")
        else:
            ok(f"Low match rate ({match_rate:.3f}) — rolling features look leak-free")

    if "days_rest" in game_df.columns:
        dr = game_df["days_rest"].dropna()
        print(f"\n  days_rest distribution:")
        print(dr.describe().to_string())
        print(f"\n  Top values:\n{dr.value_counts().head(10).to_string()}")
        if dr.min() < 0:
            warn("Negative days_rest values found")
        else:
            ok("All days_rest >= 0")

    ctx_cols = [c for c in game_df.columns if c.startswith("ctx_")]
    print(f"\n  Season-context features ({len(ctx_cols)}): {ctx_cols}")
    if ctx_cols:
        print(game_df[ctx_cols].describe().T[["mean", "std", "min", "max"]].to_string())

    info_cols = [c for c in ["age", "height", "weight", "overall_pick", "rookie_year"] if c in game_df.columns]
    print(f"\n  Player biographical features: {info_cols}")
    if info_cols:
        print(game_df[info_cols].describe().T[["mean", "std", "min", "max"]].to_string())


def inspect_model(bundle: dict):
    header("MODEL")

    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    X_train      = bundle["X_train"]
    X_test       = bundle["X_test"]

    print("  XGBRegressor parameters:")
    for k, v in sorted(model.get_params().items()):
        print(f"    {k:<30} {v}")

    print(f"\n  Train shape : {X_train.shape}")
    print(f"  Test shape  : {X_test.shape}")

    print(f"\n  Feature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"    {c}")

    imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n  Feature importance (all features):")
    print(imp.to_string(index=False))

    total = imp["importance"].sum()
    print(f"\n  Sum of importances: {total:.4f}")
    if abs(total - 1.0) >= 0.01:
        warn(f"Importances sum to {total:.4f}, expected ~1.0")
    else:
        ok("Importances sum to ~1.0")


def inspect_predictions(bundle: dict):
    header("PREDICTIONS")

    y_test = bundle["y_test"]
    y_pred = bundle["y_pred"]
    meta   = bundle["test_meta"]

    rmse          = root_mean_squared_error(y_test, y_pred)
    mae           = mean_absolute_error(y_test, y_pred)
    baseline_rmse = root_mean_squared_error(y_test, np.full_like(y_pred, y_test.mean()))
    corr          = np.corrcoef(y_test.values, y_pred)[0, 1]
    improvement   = (1 - rmse / baseline_rmse) * 100

    print(f"  Test season : {TEST_SEASON_YEAR}")
    print(f"  Test rows   : {len(y_test):,}")
    print(f"\n  RMSE          : {rmse:.3f}")
    print(f"  MAE           : {mae:.3f}")
    print(f"  Baseline RMSE : {baseline_rmse:.3f}  (always predicting mean)")
    print(f"  Improvement   : {improvement:.1f}% over baseline")
    print(f"  Pearson corr  : {corr:.4f}")

    if rmse >= baseline_rmse:
        warn("Model is worse than always predicting the mean")
    else:
        ok(f"{improvement:.1f}% improvement over naive baseline")

    print("\n  Prediction vs actual distribution:")
    print(pd.DataFrame({
        "actual":    pd.Series(y_test.values).describe(),
        "predicted": pd.Series(y_pred).describe(),
    }).to_string())

    errors = meta["error"]
    print(f"\n  Error (pred − actual) distribution:")
    print(errors.describe().to_string())
    print(f"\n  % errors within ±1 pt : {(errors.abs() <= 1).mean()*100:.1f}%")
    print(f"  % errors within ±2 pt : {(errors.abs() <= 2).mean()*100:.1f}%")
    print(f"  % errors within ±5 pt : {(errors.abs() <= 5).mean()*100:.1f}%")

    cols = ["skaterFullName", "y_actual", "y_pred", "error"] if "skaterFullName" in meta.columns \
           else ["y_actual", "y_pred", "error"]

    print(f"\n  Top 20 actual performers (test season {TEST_SEASON_YEAR}):")
    print(meta.sort_values("y_actual", ascending=False).head(20)[cols].to_string(index=False))

    print(f"\n  Top 20 model-predicted performers (test season {TEST_SEASON_YEAR}):")
    print(meta.sort_values("y_pred", ascending=False).head(20)[cols].to_string(index=False))

    print("\n  20 worst predictions by absolute error:")
    worst = meta.reindex(meta["error"].abs().sort_values(ascending=False).index).head(20)
    print(worst[cols].to_string(index=False))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("Building game dataset...")
    t0 = time.time()
    game_df = build_game_df()
    print(f"Done in {time.time() - t0:.1f}s — {len(game_df):,} rows")

    print("\nTraining model...")
    t0 = time.time()
    bundle = build_model_bundle(game_df)
    print(f"Done in {time.time() - t0:.1f}s")

    inspect_dataset(game_df)
    inspect_features(game_df)
    inspect_model(bundle)
    inspect_predictions(bundle)
