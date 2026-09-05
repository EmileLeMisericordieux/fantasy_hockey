"""
One-time MoneyPuck pull: advanced stats for every season we model.

    python scripts/fetch_moneypuck.py            # only what is missing
    python scripts/fetch_moneypuck.py --refresh  # re-download everything

Resumable: every season and kind is cached separately in data/raw/, so
re-running only fetches what is missing.

Use --refresh once the 2026-27 season is under way. MoneyPuck rewrites the
current season's file nightly, and a cached copy from October would sit there
looking valid all year.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fantasy import moneypuck
from fantasy.config import HISTORY_SEASONS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="re-download seasons already cached")
    args = ap.parse_args()

    t0 = time.time()
    total_rows = 0

    for season in HISTORY_SEASONS:
        line = [f"{season}:"]
        for kind in moneypuck.KINDS:
            t = time.time()
            df = moneypuck.season_stats(kind, season, refresh=args.refresh)
            total_rows += len(df)
            if df.empty:
                line.append(f"{kind} not published")
                continue
            n = df["playerId"].nunique()
            line.append(f"{n:>4} {kind} ({len(df):>5,} rows, {time.time() - t:.0f}s)")
        print("  ".join(line), flush=True)

    table = moneypuck.player_season_table()
    feats = moneypuck.feature_names(table)
    print(
        f"\nDone in {time.time() - t0:.0f}s -- {total_rows:,} raw rows cached.\n"
        f"Joined table: {len(table):,} player-seasons, {len(feats)} features "
        f"across {table['season'].nunique()} seasons."
    )


if __name__ == "__main__":
    main()
