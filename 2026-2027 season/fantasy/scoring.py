"""
Fantasy scoring rules -- the single source of truth.

FIRST THING TO CHECK: confirm these against the actual pool rules. If the pool
awards points for anything else (shots, PIM, blocks, power-play points,
shutouts, saves), add it here and every model, baseline and dashboard number
updates with it.
"""

from __future__ import annotations

import pandas as pd

from .config import DEFENCE_POS, GOALIE_POS


# ─── Rules ─────────────────────────────────────────────────────────────────────
# Edit these, not the functions below.

RULES = {
    "forward": {"goal": 2.0, "assist": 1.0},
    "defence": {"goal": 3.0, "assist": 2.0},
    "goalie": {"win": 3.0, "shutout": 4.0, "otl": 0.0},
    "bonus": {"hat_trick_f": 3.0, "hat_trick_d": 5.0},
}


def score_skater_games(df: pd.DataFrame) -> pd.Series:
    """Vectorised per-game fantasy points for skaters."""
    goals = pd.to_numeric(df["goals"], errors="coerce").fillna(0)
    assists = pd.to_numeric(df["assists"], errors="coerce").fillna(0)
    is_d = df["positionCode"].isin(DEFENCE_POS)

    goal_val = pd.Series(RULES["forward"]["goal"], index=df.index).mask(
        is_d, RULES["defence"]["goal"]
    )
    assist_val = pd.Series(RULES["forward"]["assist"], index=df.index).mask(
        is_d, RULES["defence"]["assist"]
    )

    pts = goals * goal_val + assists * assist_val
    pts += ((goals >= 3) & ~is_d).astype(float) * RULES["bonus"]["hat_trick_f"]
    pts += ((goals >= 3) & is_d).astype(float) * RULES["bonus"]["hat_trick_d"]
    return pts.astype(float)


def score_goalie_games(df: pd.DataFrame) -> pd.Series:
    """Per-game fantasy points for goalies."""
    if "decision" in df.columns:
        wins = (df["decision"].astype(str).str.upper() == "W").astype(float)
    elif "wins" in df.columns:
        wins = pd.to_numeric(df["wins"], errors="coerce").fillna(0)
    else:
        wins = pd.Series(0.0, index=df.index)

    pts = wins * RULES["goalie"]["win"]

    if RULES["goalie"]["shutout"] and "shutouts" in df.columns:
        pts += pd.to_numeric(df["shutouts"], errors="coerce").fillna(0) * RULES["goalie"]["shutout"]

    return pts.astype(float)


def add_fantasy_points(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `fantasy_points` column, dispatching on position."""
    df = df.copy()
    is_goalie = df["positionCode"].isin(GOALIE_POS)

    pts = pd.Series(0.0, index=df.index)
    if (~is_goalie).any():
        pts.loc[~is_goalie] = score_skater_games(df.loc[~is_goalie])
    if is_goalie.any():
        pts.loc[is_goalie] = score_goalie_games(df.loc[is_goalie])

    df["fantasy_points"] = pts
    return df
