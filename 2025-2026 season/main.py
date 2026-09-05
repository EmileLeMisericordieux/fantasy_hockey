from feature_engineering import FantasyDataEngineer
from helper_functions import quick_profile

import pandas as pd
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import time


# ─── CONFIG ────────────────────────────────────────────────────────────────────

TRAIN_SEASONS = [f"{y}{y+1}" for y in range(2020, 2025)]
SCORE_SEASONS = [f"{y}{y+1}" for y in range(2024, 2026)]

# Drop rows where rolling features are based on too few past games (cold-start)
MIN_GAMES_BEFORE_PREDICT = 5

# Raw per-game stats that would leak directly into the target
RAW_STAT_COLS = [
    "goals", "assists", "points", "shots", "toi", "timeOnIce", "toi_minutes",
    "powerPlayGoals", "powerPlayAssists", "shortHandedGoals", "shortHandedAssists",
    "plusMinus", "pim", "decision", "homeRoadFlag", "fantasy_points", "p60",
]

# Metadata / ID columns that carry no predictive signal
META_COLS = [
    "playerId", "seasonId", "gameId", "season", "season_start_year", "gameDate",
    "skaterFullName", "lastName", "teamAbbrevs", "opponentTeamAbbrev",
]

TARGET = "season_fantasy_remaining"   # model predicts points still to earn
TOTAL_COL = "season_fantasy_total"    # kept for evaluation / reference

# Categorical columns to one-hot encode
CAT_COLS_CANDIDATES = ["positionCode", "shootsCatches"]


# ─── DATA FETCHING ──────────────────────────────────────────────────────────────

engineer = FantasyDataEngineer(seasons=TRAIN_SEASONS, limit=100, debug=False)

t0 = time.time()
game_df = engineer.build_game_dataset()
print(f"\nBuild complete in {time.time() - t0:.1f}s — {len(game_df):,} game rows\n")

print(quick_profile(game_df))


# ─── FEATURE PREP ──────────────────────────────────────────────────────────────

# Remove cold-start rows (rolling features unreliable in first N games of season)
game_df = game_df[game_df["games_in_season"] >= MIN_GAMES_BEFORE_PREDICT].copy()

CAT_COLS = [c for c in CAT_COLS_CANDIDATES if c in game_df.columns]
df_model = pd.get_dummies(game_df, columns=CAT_COLS, drop_first=True)

DROP_COLS = [c for c in RAW_STAT_COLS + META_COLS + [TARGET, TOTAL_COL] if c in df_model.columns]
FEATURE_COLS = [
    c for c in df_model.columns
    if c not in DROP_COLS and df_model[c].dtype != object
]

print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")


# ─── WALK-FORWARD CV BY SEASON ─────────────────────────────────────────────────
# Train on all games up to season X, test on games in season X+1.

unique_seasons = sorted(df_model["season_start_year"].unique())
results = []

for cutoff in unique_seasons[:-1]:
    train = df_model[df_model["season_start_year"] <= cutoff].dropna(subset=[TARGET])
    test  = df_model[df_model["season_start_year"] == cutoff + 1].dropna(subset=[TARGET])

    if len(test) == 0:
        continue

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test,  y_test  = test[FEATURE_COLS],  test[TARGET]

    # Align columns in case get_dummies produced different dummies per split
    X_test = X_test.reindex(columns=FEATURE_COLS, fill_value=0)

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
    pred_remaining = model.predict(X_test)
    pred_total = test["cum_fantasy_so_far"].values + pred_remaining
    actual_total = test[TOTAL_COL].values if TOTAL_COL in test.columns else y_test.values + test["cum_fantasy_so_far"].values
    rmse = root_mean_squared_error(actual_total, pred_total)
    results.append({"train_upto": cutoff, "test_season": cutoff + 1, "rmse": rmse})
    print(f"  cutoff={cutoff}  test_season={cutoff+1}  rmse={rmse:.3f}")

print("\nWalk-forward RMSE:")
print(pd.DataFrame(results).to_string(index=False))


# ─── FEATURE IMPORTANCE ────────────────────────────────────────────────────────

importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop 15 features:")
print(importance_df.head(15).to_string(index=False))


# ─── SCORE: PREDICT NEXT GAME FOR EACH ACTIVE PLAYER ──────────────────────────
# Strategy: take each player's most recent game row as their feature vector.
# The rolling features already encode recent form — this represents their
# "current state" from which we predict the next game.
#
# NOTE: To incorporate injuries, filter out injured players here before scoring,
# using an external source (e.g. Rotowire or the NHL injury report API).

score_engineer = FantasyDataEngineer(seasons=SCORE_SEASONS, limit=100, debug=False)

t0 = time.time()
score_df = score_engineer.build_game_dataset()
print(f"\nScore data ready in {time.time() - t0:.1f}s — {len(score_df):,} rows")

score_df = score_df[score_df["games_in_season"] >= MIN_GAMES_BEFORE_PREDICT].copy()
score_model = pd.get_dummies(score_df, columns=CAT_COLS, drop_first=True)

# Latest game per player = most current rolling feature state
latest_rows = (
    score_model
    .sort_values("gameDate")
    .groupby("playerId", as_index=False)
    .last()
)

X_score = latest_rows.reindex(columns=FEATURE_COLS, fill_value=0)
latest_rows = latest_rows.copy()
latest_rows["predicted_pts"] = model.predict(X_score)

id_cols = [c for c in ["playerId", "skaterFullName", "season", "predicted_pts"] if c in latest_rows.columns]
output = latest_rows[id_cols].copy()
output = output.sort_values("predicted_pts", ascending=False).reset_index(drop=True)

print("\nTop 30 predicted fantasy scores for next game:")
print(output.head(30).to_string(index=False))

output.to_csv("predicted_scores_per_game.csv", index=False)
print("\nSaved → predicted_scores_per_game.csv")
