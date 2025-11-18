from feature_engineering import FantasyDataEngineer
from helper_functions import quick_profile
from nhlpy import NHLClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import time


#------------------------- DATA FETCHING & ENGINEERING -------------------------#
seasons = [f"{y}{y+1}" for y in range(2020, 2025)]
engineer = FantasyDataEngineer(seasons=seasons, limit=100, debug=False)
time_start = time.time()
df = engineer.build_dataset()
time_end = time.time()
print(f"Data fetched and engineered in {time_end - time_start:.2f} seconds.")

#------------------------- DATA PROFILING -------------------------#
profile_df = quick_profile(df)
print(profile_df)



df_model = pd.get_dummies(df, columns=["positionCode", "shootsCatches"], drop_first=True)

unique_seasons = sorted(df_model["season_start_year"].unique())

results = []
for cutoff in unique_seasons[:-1]:  # last season cannot be tested
    train = df_model[df_model["season_start_year"] <= cutoff]
    test  = df_model[df_model["season_start_year"] == cutoff + 1]

    # Drop rows with missing next-season score
    train = train.dropna(subset=["target"])
    test  = test.dropna(subset=["target"])

    if len(test) == 0:
        continue  # skip missing seasons

    # Build X/y
    X_train = train.drop(columns=["target", "season_start_year", "season", 'lastName', 'playerId', 'seasonId', 'skaterFullName', "teamAbbrevs"])
    y_train = train["target"]

    X_test  = test.drop(columns=["target", "season_start_year", "season", 'lastName', 'playerId', 'seasonId', 'skaterFullName', "teamAbbrevs"])
    y_test  = test["target"]

    # train model
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    # predict on season cutoff+1 only
    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    results.append({"train_upto": cutoff, "test_season": cutoff+1, "rmse": rmse})

results_df = pd.DataFrame(results)
print(results_df)

comparison = test.copy()
comparison["y_test"] = y_test.values
comparison["y_pred"] = y_pred

# Keep only relevant columns
comparison = comparison[[
    "skaterFullName",
    "season",
    "y_test",
    "y_pred"
]]

print(comparison.sort_values("y_test", ascending=False).head(20))
print(comparison.sort_values("y_pred", ascending=False).head(20))

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

top10 = importance_df.sort_values("importance", ascending=False).head(10)

print(top10)

#------------------------- SCORING ------------------------#

seasons = [f"{y}{y+1}" for y in range(2024, 2026)]
engineer = FantasyDataEngineer(seasons=seasons, limit=100, debug=False)
time_start = time.time()
score_df = engineer.build_dataset()
time_end = time.time()
print(f"Data fetched and engineered in {time_end - time_start:.2f} seconds.")

df_model = pd.get_dummies(score_df, columns=["positionCode", "shootsCatches"], drop_first=True)
score_final_df  = df_model.drop(columns=["target", "season_start_year", "season", 'lastName', 'playerId', 'seasonId', 'skaterFullName', "teamAbbrevs"])
y_test  = df_model["target"]
y_pred = model.predict(score_final_df)

comparison = df_model.copy()
comparison["y_test"] = y_test.values
comparison["y_pred"] = y_pred

comparison = comparison[[
    "skaterFullName",
    'playerId',
    "season",
    "y_test",
    "y_pred"
]]

print(comparison.sort_values("y_test", ascending=False).head(20))
print(comparison.sort_values("y_pred", ascending=False).head(100))

comparison.sort_values("y_pred", ascending=False).to_csv("predicted_scores.csv", index=False)


for col in comparison.columns:
    print(col)
