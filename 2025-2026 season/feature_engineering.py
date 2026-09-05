from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
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
    workers: int = 8          # parallel threads for API calls

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
        # 1) cache API calls — fetch all players in parallel
        unique_ids = df["playerId"].unique()

        def _fetch_one(pid):
            return pid, self.client.stats.player_career_stats(player_id=pid)

        player_cache = {}
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(_fetch_one, pid): pid for pid in unique_ids}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Player info"):
                try:
                    pid, info = future.result()
                    player_cache[pid] = info
                except Exception as e:
                    pid = futures[future]
                    player_cache[pid] = {}
                    if self.debug:
                        print(f"    → player info error pid={pid}: {e}")

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
    # Game-level pipeline
    # ------------------------------------------------------------------ #

    def fetch_all_game_logs(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Fetch per-game logs for every player-season in df (or self.raw_df).
        Returns a long DataFrame: one row per player-game.
        """
        if df is None:
            if self.raw_df is None:
                raise ValueError("No raw data. Call fetch_raw_data() first.")
            df = self.raw_df

        combos = df[["playerId", "season", "positionCode"]].drop_duplicates().reset_index(drop=True)
        all_logs: List[dict] = []

        def _fetch_combo(row):
            pid = int(row.playerId)
            season = row.season
            pos = row.positionCode
            try:
                logs = self.client.stats.player_game_log(
                    player_id=pid, season_id=season, game_type=int(self.game_type)
                )
                for g in logs:
                    g["playerId"] = pid
                    g["season"] = season
                    g["positionCode"] = pos
                return logs
            except Exception as e:
                if self.debug:
                    print(f"    → error player={pid} season={season}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(_fetch_combo, row): row
                for row in combos.itertuples(index=False)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Game logs"):
                all_logs.extend(future.result())

        game_df = pd.DataFrame(all_logs)

        # Merge player names from season summary
        if "skaterFullName" in df.columns:
            name_map = df[["playerId", "skaterFullName"]].drop_duplicates("playerId")
            game_df = game_df.merge(name_map, on="playerId", how="left")

        # Parse TOI string "MM:SS" → float minutes
        toi_col = next((c for c in ["toi", "timeOnIce"] if c in game_df.columns), None)
        if toi_col:
            game_df["toi_minutes"] = game_df[toi_col].apply(self._parse_toi)

        return game_df

    def compute_game_fantasy(self, game_df: pd.DataFrame) -> pd.DataFrame:
        """Add per-game fantasy_points column (hat tricks inferred from goals >= 3)."""
        game_df = game_df.copy()
        game_df["fantasy_points"] = game_df.apply(self._fantasy_score_game_row, axis=1)
        return game_df

    def add_rolling_features(
        self,
        game_df: pd.DataFrame,
        windows: List[int] = [3, 5, 10, 20],
    ) -> pd.DataFrame:
        """
        Add rolling-window averages per player sorted by game date.
        Uses shift(1) before rolling so each row only sees past games (no leakage).
        Also adds: days_rest, games_in_season, is_home.
        """
        game_df = game_df.copy()
        game_df["gameDate"] = pd.to_datetime(game_df["gameDate"])
        game_df = game_df.sort_values(["playerId", "gameDate"]).reset_index(drop=True)

        # Points per 60 minutes — normalises production by ice time
        if "goals" in game_df.columns and "assists" in game_df.columns and "toi_minutes" in game_df.columns:
            game_df["p60"] = (
                (game_df["goals"] + game_df["assists"])
                / game_df["toi_minutes"].clip(lower=1)
            ) * 60

        roll_cols = [
            c for c in ["fantasy_points", "goals", "assists", "shots", "toi_minutes", "plusMinus", "p60"]
            if c in game_df.columns
        ]

        for w in windows:
            for col in roll_cols:
                game_df[f"roll{w}_{col}"] = (
                    game_df.groupby("playerId")[col]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )

        # Days since last game (NaN on first game → fill with 7)
        game_df["days_rest"] = (
            game_df.groupby("playerId")["gameDate"]
            .transform(lambda x: x.diff().dt.days)
            .fillna(7)
        )

        # How many games this player has played so far in the season (0 = first)
        game_df["games_in_season"] = game_df.groupby(["playerId", "season"]).cumcount()

        # Home/away flag
        if "homeRoadFlag" in game_df.columns:
            game_df["is_home"] = (game_df["homeRoadFlag"] == "H").astype(int)

        return game_df

    def _merge_season_context(self, season_df: pd.DataFrame, game_df: pd.DataFrame) -> pd.DataFrame:
        """Attach season-level rate stats to every game row as contextual features."""
        candidate_ctx = [
            "shootingPct", "pointsPerGame", "powerPlayPoints",
            "gamesPlayed", "faceoffWinPct", "timeOnIcePerGame", "savePct",
        ]
        ctx_cols = ["playerId", "season"] + [c for c in candidate_ctx if c in season_df.columns]
        ctx = season_df[ctx_cols].copy()
        ctx = ctx.rename(columns={c: f"ctx_{c}" for c in ctx_cols if c not in ["playerId", "season"]})
        return game_df.merge(ctx, on=["playerId", "season"], how="left")

    def build_game_dataset(self) -> pd.DataFrame:
        """
        Full game-level pipeline:
          1. fetch_raw_data()              → season summaries (enumerates player-seasons)
          2. fetch_all_game_logs()         → one row per player-game
          3. compute_game_fantasy()        → per-game fantasy_points
          4. _add_season_total()           → season_fantasy_total target column
          5. add_rolling_features()        → rolling-window features (no leakage)
          6. add_cumulative_features()     → season-to-date cumulative features
          7. _merge_season_context()       → season-level rate stats as context
          8. fetch_player_info_to_df()     → height, weight, age, draft info
        """
        print("Fetching season summaries...")
        raw = self.fetch_raw_data()

        n_combos = raw[["playerId", "season"]].drop_duplicates().shape[0]
        print(f"Fetching game logs for {n_combos} player-seasons...")
        game_df = self.fetch_all_game_logs(raw)

        print("Computing per-game fantasy scores...")
        game_df = self.compute_game_fantasy(game_df)

        print("Adding rolling features...")
        game_df = self.add_rolling_features(game_df)

        print("Adding cumulative features...")
        game_df = self.add_cumulative_features(game_df)

        print("Adding season total target...")
        game_df = self._add_season_total(game_df)

        print("Merging season context...")
        game_df = self._merge_season_context(raw, game_df)

        game_df["season_start_year"] = game_df["season"].astype(str).str[:4].astype(int)

        print("Fetching player biographical info...")
        game_df = self.fetch_player_info_to_df(game_df)

        self.dataset = game_df
        return game_df

    def add_cumulative_features(
        self, game_df: pd.DataFrame, season_length: int = 82
    ) -> pd.DataFrame:
        """
        Add season-to-date cumulative features that let the model anchor its
        prediction to what has already happened this season.

        All features use shift(1) so each row only sees games *before* the
        current one — no leakage into the target.

        Features added
        --------------
        cum_fantasy_so_far  : total fantasy points earned before this game
        games_remaining     : estimated games left (season_length - games_in_season)
        season_progress     : fraction of season completed (0 → 1)
        projected_pace      : (cum_so_far / games_played) * season_length
                              — naive linear extrapolation of current pace
        points_needed_pace  : how many points per remaining game are needed
                              to match projected_pace from scratch (0 at start,
                              equals current per-game rate once converged)
        """
        game_df = game_df.copy()
        game_df = game_df.sort_values(["playerId", "season", "gameDate"]).reset_index(drop=True)

        # Cumulative points BEFORE this game
        game_df["cum_fantasy_so_far"] = (
            game_df.groupby(["playerId", "season"])["fantasy_points"]
            .transform(lambda x: x.shift(1).cumsum().fillna(0))
        )

        # games_in_season is 0-indexed cumcount → games played before this game
        games_played = game_df["games_in_season"].clip(lower=1)
        games_remaining = (season_length - game_df["games_in_season"]).clip(lower=0)

        game_df["games_remaining"] = games_remaining
        game_df["season_progress"]  = game_df["games_in_season"] / season_length

        # Naive pace extrapolation: if I keep scoring at my current rate...
        game_df["projected_pace"] = (
            game_df["cum_fantasy_so_far"] / games_played * season_length
        )

        # Points still needed per game to hit projected_pace
        game_df["points_needed_pace"] = np.where(
            games_remaining > 0,
            (game_df["projected_pace"] - game_df["cum_fantasy_so_far"]) / games_remaining,
            0.0,
        )

        return game_df

    def _add_season_total(self, game_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add season_fantasy_total and season_fantasy_remaining.

        season_fantasy_total     : full-season sum (kept for reference / plotting)
        season_fantasy_remaining : points yet to be earned AFTER this game
                                   = season_fantasy_total - cum_fantasy_so_far
                                   This is the model TARGET — predicting remaining
                                   points makes underestimation structurally impossible
                                   because the model only needs to predict ≥ 0.

        NOTE: cum_fantasy_so_far must already be present (call add_cumulative_features first).
        """
        game_df = game_df.copy()
        game_df["season_fantasy_total"] = (
            game_df.groupby(["playerId", "season"])["fantasy_points"].transform("sum")
        )
        if "cum_fantasy_so_far" in game_df.columns:
            game_df["season_fantasy_remaining"] = (
                game_df["season_fantasy_total"] - game_df["cum_fantasy_so_far"]
            ).clip(lower=0)
        return game_df

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

    @staticmethod
    def _parse_toi(toi_str) -> float:
        """Convert 'MM:SS' string to total minutes as float. Returns 0.0 on error."""
        if pd.isna(toi_str) or not isinstance(toi_str, str):
            return 0.0
        try:
            m, s = toi_str.split(":")
            return int(m) + int(s) / 60.0
        except (ValueError, AttributeError):
            return 0.0

    def _fantasy_score_game_row(self, row: pd.Series) -> float:
        """
        Per-game fantasy score.
        Hat tricks are inferred from goals >= 3 (no separate column needed).
        Goalie wins are inferred from decision == 'W'.
        """
        pos = row.get("positionCode", None)
        goals = int(row.get("goals", 0) or 0)
        assists = int(row.get("assists", 0) or 0)
        hat = 1 if goals >= 3 else 0

        if pos in GOALIE_POS:
            return 3.0 if row.get("decision", "") == "W" else 0.0

        if pos in DEFENCE_POS:
            return float(3 * goals + 2 * assists + 3 * hat)

        return float(2 * goals + 1 * assists + 3 * hat)

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
