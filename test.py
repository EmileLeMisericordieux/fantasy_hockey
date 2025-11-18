from nhlpy import NHLClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


client = NHLClient()

def fetch_player_info(player_id: int, season: int) -> dict:
    """Fetch and return player career stats for a given player ID."""
    player_info = {}
    info = client.stats.player_career_stats(player_id=player_id)
    player_info["height"] = info["heightInCentimeters"]
    player_info["weight"] = info["weightInKilograms"]
    player_info["age"] = season - int(info["birthDate"][:4])
    player_info["rookie_year"] = [1 if (season == int(info["draftDetails"]["year"])) else 0][0]
    player_info["overall_pick"] = info["draftDetails"]["overallPick"]

    return player_info

def fetch_player_info_to_df(df: pd.DataFrame) -> pd.DataFrame:
    player_info_df = df.apply(
        lambda row: pd.Series(fetch_player_info(row["playerId"], row["season"])),
        axis=1
    )

    # 2) Join the new columns back to the original df
    df = df.join(player_info_df)
    return df


df = pd.DataFrame({
    "playerId": [8468208, 8478403],  # Example player IDs
    "season": [2015, 2015]
})

fetch_player_info_to_df(df)  # Example player ID




#-------------------------------
import json

info = client.stats.player_career_stats(player_id=8468208)
with open("output.json", "w") as f:
    json.dump(info, f, indent=4, ensure_ascii=False)

info = client.stats.player_game_log(player_id=8478402, season_id="20242025", game_type=2)
with open("output.json", "w") as f:
    json.dump(info, f, indent=4, ensure_ascii=False)

def count_hat_tricks(client, player_id: int, season: int) -> int:
    logs = client.stats.player_game_log(player_id=player_id, season_id=season, game_type=2)

    hat_tricks = 0
    for game in logs:
        if game.get("goals", 0) >= 3:
            hat_tricks += 1

    return hat_tricks

print(count_hat_tricks(client, 8484144, "20252026"))  # Example player ID