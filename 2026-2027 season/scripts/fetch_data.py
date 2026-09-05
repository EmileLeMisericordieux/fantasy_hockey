"""
One-time data pull. Run this before the workshop so nobody waits on the API.

    python scripts/fetch_data.py

Resumable: every season is cached separately in data/raw/, so re-running only
fetches what is missing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fantasy import data
from fantasy.config import HISTORY_SEASONS


def main() -> None:
    t0 = time.time()
    total_rows = 0

    for season in HISTORY_SEASONS:
        t = time.time()
        df = data.game_logs(season)
        total_rows += len(df)
        n_goalies = (df["positionCode"] == "G").sum() if not df.empty else 0
        print(
            f"{season}: {len(df):>7,} rows "
            f"({df['playerId'].nunique():>4} players, {n_goalies:>5,} goalie rows) "
            f"[{time.time() - t:.0f}s]",
            flush=True,
        )

    print("\nFetching player biographies...", flush=True)
    all_games = data.load_seasons(HISTORY_SEASONS)
    bios = data.player_bios(all_games["playerId"].unique())

    print(
        f"\nDone in {time.time() - t0:.0f}s -- "
        f"{total_rows:,} player-game rows, {len(bios):,} player bios."
    )


if __name__ == "__main__":
    main()
