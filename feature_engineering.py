from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import numpy as np
from nhlpy import NHLClient
from nhlpy.api.query.builder import QueryBuilder, QueryContext
from nhlpy.api.query.filters.season import SeasonQuery
from nhlpy.api.query.filters.game_type import GameTypeQuery


FORWARD_POS = {"C", "LW", "RW", "L", "R", "F"}
DEFENCE_POS = {"D"}
GOALIE_POS = {"G"}


def make_season_list(start_year: int = 2010, end_year: int = 2024) -> List[str]:
    """Return list of NHL seasons in 'YYYYYYYY+1' format."""
    return [f"{year}{year + 1}" for year in range(start_year, end_year + 1)]


@dataclass
class FantasyDataEngineer:
    """
    End-to-end builder for a historical NHL player dataset
    with fantasy scores and next-season targets.

    Steps:
      1. fetch_raw_data()      -> pulls skater stats for all seasons
      2. compute_fantasy()     -> adds fantasy_points column
      3. add_next_season_target() -> adds target column (next year's fantasy_points)
      4. build_dataset()       -> runs everything and returns final DataFrame
    """

    seasons: List[str] = field(default_factory=lambda: make_season_list(2010, 2024))
    limit: int = 100          # pagination batch size
    game_type: str = "2"      # "2" = regular season
    debug: bool = False

    client: NHLClient = field(default=None, init=False)
    raw_df: Optional[pd.DataFrame] = field(default=None, init=False)
    dataset: Optional[pd.DataFrame] = field(default=None, init=False)

    # ------------------------------------------------------------------ #
    # Core public API
    # ------------------------------------------------------------------ #
    def fetch_raw_data(self) -> pd.DataFrame:
        """
        Fetch summary statistics for skaters for all configured seasons.
        Handles pagination and concatenates all seasons into one DataFrame.
        """
        if self.client is None:
            self.client = NHLClient(debug=self.debug)

        all_dfs: List[pd.DataFrame] = []

        for season in self.seasons:
            season_df = self._load_summary_statistics_for_skaters(
                season_start=season,
                season_end=season,
            )
            season_df["season"] = season
            all_dfs.append(season_df)

        self.raw_df = pd.concat(all_dfs, ignore_index=True)
        return self.raw_df
    
    def add_hat_trick_counts(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add a 'hat_tricks' column to the given DataFrame (or self.raw_df),
        counting games with >= 3 goals per player-season.
        """
        if df is None:
            if self.raw_df is None:
                raise ValueError("No raw data available. Call fetch_raw_data() first.")
            df = self.raw_df

        # we assume df has 'playerId' and 'season' columns
        required_cols = {"playerId", "season"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns in df: {missing}")

        # unique player-season combos to avoid repeating the same API calls
        combos = df[["playerId", "season"]].drop_duplicates().reset_index(drop=True)

        combos["hat_tricks"] = combos.apply(
            lambda row: self._count_hat_tricks(
                int(row["playerId"]), row["season"]
            ),
            axis=1,
        )

        # merge back to original df
        df = df.merge(combos, on=["playerId", "season"], how="left")
        return df

    def compute_fantasy(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add fantasy_points column to the given DataFrame (or self.raw_df).
        """
        if df is None:
            if self.raw_df is None:
                raise ValueError("No raw data available. Call fetch_raw_data() first.")
            df = self.raw_df

        df_scored = self._add_fantasy_score(df)
        return df_scored

    def add_next_season_target(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add 'target' column = next season's fantasy_points for the same player.
        Drops rows with no next-season target.

        Also adds 'season_start_year' and sorts by it.
        """
        if df is None:
            if self.raw_df is None:
                raise ValueError("No raw data available. Call fetch_raw_data() first.")
            df = self.raw_df

        if "fantasy_points" not in df.columns:
            raise ValueError("fantasy_points not found. Call compute_fantasy() first.")

        df = df.copy()
        df = df.sort_values(["playerId", "season"])

        # target: next season's fantasy_points
        df["target"] = df.groupby("playerId")["fantasy_points"].shift(-1)

        # drop last season per player (no next-season target)
        df = df.dropna(subset=["target"])

        # parse season start year from 'YYYYYYYY+1'
        df["season"] = df["season"].astype(str)
        df["season_start_year"] = df["season"].str[:4].astype(int)

        df = df.sort_values(["season_start_year"]).reset_index(drop=True)

        return df
    
    def fetch_player_info(self, player_id: int, season: int) -> dict:
        """Fetch and return player career stats for a given player ID."""
        player_info = {}
        info = self.client.stats.player_career_stats(player_id=player_id)
        player_info["height"] = info["heightInCentimeters"]
        player_info["weight"] = info["weightInKilograms"]
        player_info["age"] = season - int(info["birthDate"][:4])

        try:
            player_info["rookie_year"] = [1 if (season == int(info["draftDetails"]["year"])) else 0][0]
        except:
            player_info["rookie_year"] = np.nan
        
        try:
            player_info["overall_pick"] = info["draftDetails"]["overallPick"]
        except:
            player_info["overall_pick"] = np.nan

        return player_info

    def fetch_player_info_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) cache API calls
        unique_ids = df["playerId"].unique()
        player_cache = {pid: self.client.stats.player_career_stats(player_id=pid)
                        for pid in unique_ids}

        # 2) build info rows
        def build_row(row):
            pid = row["playerId"]
            season = row["season_start_year"]
            info = player_cache[pid]

            birth = info.get("birthDate")
            draft = info.get("draftDetails") or {}

            return pd.Series({
                "height": info.get("heightInCentimeters", np.nan),
                "weight": info.get("weightInKilograms", np.nan),
                "age": season - int(birth[:4]) if birth else np.nan,
                "rookie_year": int(season == int(draft.get("year"))) if draft.get("year") else np.nan,
                "overall_pick": draft.get("overallPick", np.nan),
            })

        player_info_df = df.apply(build_row, axis=1)
        df = df.join(player_info_df)

        return df

    def build_dataset(self) -> pd.DataFrame:
        """
        Convenience method: runs the full pipeline and returns a clean dataset
        ready for modeling.
        """
        raw = self.fetch_raw_data()
        raw_with_hat = self.add_hat_trick_counts(raw)
        scored = self.compute_fantasy(raw_with_hat)
        final = self.add_next_season_target(scored)
        final = self.fetch_player_info_to_df(final)

        self.dataset = final
        return final

    # ------------------------------------------------------------------ #
    # Internal helpers: NHL API + pagination
    # ------------------------------------------------------------------ #
    def _build_query_context(self, season_start: str, season_end: str) -> QueryContext:
        """
        Build NHL query context for given season range and game type.
        """
        filters = [
            SeasonQuery(season_start=season_start, season_end=season_end),
            GameTypeQuery(game_type=self.game_type),
        ]
        context: QueryContext = QueryBuilder().build(filters=filters)
        return context

    def _load_summary_statistics_for_skaters(
        self,
        season_start: str,
        season_end: str,
    ) -> pd.DataFrame:
        """
        Low-level: pull all pages of skater summary stats for a season range.
        Returns a DataFrame.
        """
        context = self._build_query_context(season_start, season_end)
        all_data = []
        start = 0

        while True:
            response = self.client.stats.skater_stats_with_query_context(
                report_type="summary",
                query_context=context,
                aggregate=False,
                limit=self.limit,
                start=start,
            )

            total = response["total"]
            batch_data = response["data"]
            all_data.extend(batch_data)

            if len(all_data) >= total:
                break

            start += self.limit

        return pd.DataFrame(all_data)

    # ------------------------------------------------------------------ #
    # Internal helpers: fantasy score
    # ------------------------------------------------------------------ #

    def _count_hat_tricks(self, player_id: int, season: int) -> int:
        logs = self.client.stats.player_game_log(player_id=player_id, season_id=season, game_type=2)

        hat_tricks = 0
        for game in logs:
            if game.get("goals", 0) >= 3:
                hat_tricks += 1

        return hat_tricks
    
    def _fantasy_score_row(self, row: pd.Series, win_col: str = "wins") -> float:
        """
        Calculate fantasy score for a single player row.

        Rules:
          - Forwards:  goal = 2, assist = 1
          - Defence:   goal = 3, assist = 2
          - Goalies:   win  = 3  (if wins column is available)

        Position is read from 'position' or 'positionCode'.
        """
        # position can be 'position' or 'positionCode'
        pos = row.get("position", None)
        if pos is None:
            pos = row.get("positionCode", None)

        # Defaults
        goals = row.get("goals", 0) or 0
        assists = row.get("assists", 0) or 0
        hat = row.get("hat_tricks", 0) or 0

        # Goalie scoring
        if pos in GOALIE_POS:
            wins = row.get(win_col, None)
            if wins is None:
                wins = row.get("winsInSeason", 0) or 0
            return 3 * wins

        # Defence scoring
        if pos in DEFENCE_POS:
            return 3 * goals + 2 * assists + 3 * hat

        # Forwards / everything else treated as forward
        return 2 * goals + 1 * assists + 3 * hat

    def _add_fantasy_score(self, df: pd.DataFrame, win_col: Optional[str] = None) -> pd.DataFrame:
        """
        Return a copy of df with a 'fantasy_points' column added.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain per-player season stats.
        win_col : str or None
            Name of the goalie wins column. If None, tries 'wins' then 'winsInSeason'.
        """
        if win_col is None:
            if "wins" in df.columns:
                win_col = "wins"
            elif "winsInSeason" in df.columns:
                win_col = "winsInSeason"
            else:
                win_col = None  # handled in _fantasy_score_row

        df = df.copy()
        df["fantasy_points"] = df.apply(
            lambda row: self._fantasy_score_row(row, win_col=win_col),
            axis=1,
        )
        return df
