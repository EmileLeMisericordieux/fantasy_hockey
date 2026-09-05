"""Central configuration. Everything that might change lives here."""

from __future__ import annotations

from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

for _d in (RAW, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)


# ─── Seasons ───────────────────────────────────────────────────────────────────

def season_id(start_year: int) -> str:
    """2023 -> '20232024'."""
    return f"{start_year}{start_year + 1}"


# Seasons we pull game logs for. 2026-27 has not started yet, so history ends
# with the completed 2025-26 season.
HISTORY_START = 2015
HISTORY_END = 2025          # 2025-26 is complete and is our validation season
TARGET_SEASON_START = 2026  # the season we are drafting for

HISTORY_YEARS = list(range(HISTORY_START, HISTORY_END + 1))
HISTORY_SEASONS = [season_id(y) for y in HISTORY_YEARS]
TARGET_SEASON = season_id(TARGET_SEASON_START)

# Real number of regular-season games per team. Assuming 82 across the board
# breaks on the two COVID seasons, which throws off anything per-game.
SEASON_LENGTH = {
    "20192020": 70,   # halted 2020-03-12, teams played 68-71
    "20202021": 56,   # shortened season
}
DEFAULT_SEASON_LENGTH = 82


def season_length(season: str) -> int:
    return SEASON_LENGTH.get(str(season), DEFAULT_SEASON_LENGTH)


# Seasons distorted enough that per-season totals are not comparable.
# Useful as a feature (`is_short_season`) and as a filter.
SHORT_SEASONS = set(SEASON_LENGTH)


# ─── Positions ─────────────────────────────────────────────────────────────────

FORWARD_POS = {"C", "L", "R", "LW", "RW", "F"}
DEFENCE_POS = {"D"}
GOALIE_POS = {"G"}


# ─── API ───────────────────────────────────────────────────────────────────────

GAME_TYPE_REGULAR = 2
API_WORKERS = 8      # threads for game-log fetching
PAGE_LIMIT = 100     # pagination size for summary endpoints
