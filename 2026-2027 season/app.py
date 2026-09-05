"""
Fantasy hockey dashboard, 2026-27.

    python -m streamlit run app.py

Use `python -m streamlit`, not bare `streamlit`. With several Python installs on
one machine, bare `streamlit` runs whichever PATH finds first, which may not be
the one holding these packages.

Three tabs:
  Draft Board    A working first pass. Improving it is the point.
  Model Lab      Compare approaches on the same walk-forward split.
  Season Tracker Empty until opening night -- see the TODO list in the tab.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from fantasy import data, datasets, evaluate, models
from fantasy.config import DEFENCE_POS, FORWARD_POS, GOALIE_POS, TARGET_SEASON

st.set_page_config(page_title="Fantasy Hockey 2026-27", layout="wide")

ROSTER_FILE = Path("roster.json")
MIN_GAMES = 20
TARGET = "fp_per82"
POSITION_GROUPS = {
    "A": FORWARD_POS,
    "D": DEFENCE_POS,
    "G": GOALIE_POS,
}


# ─── Data ──────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Building history…")
def get_history() -> pd.DataFrame:
    df = datasets.draft_table()
    return df[(df.seasons_of_history >= 1) & (df.games_played >= MIN_GAMES)].copy()


@st.cache_data(show_spinner="Building 2026-27 candidates…")
def get_upcoming() -> pd.DataFrame:
    df = datasets.upcoming_draft_table()
    return df[df.seasons_of_history >= 1].copy()


@st.cache_data(show_spinner="Projecting…")
def get_projections(target: str) -> pd.DataFrame:
    hist, up = get_history(), get_upcoming()
    feats = datasets.feature_columns(hist, task="draft")
    up = up.copy()
    up["projection"] = models.make_gbm()(hist, up, feats, target)
    return up.sort_values("projection", ascending=False).reset_index(drop=True)


def load_roster() -> list[int]:
    return json.loads(ROSTER_FILE.read_text()) if ROSTER_FILE.exists() else []


def save_roster(ids: list[int]) -> None:
    ROSTER_FILE.write_text(json.dumps(ids))


def add_edge(df: pd.DataFrame, slots: dict[str, int], n_teams: int) -> pd.DataFrame:
    """
    "Edge" = how much better a player is than the worst starter at his position.

    Why not just draft the highest projection? Because in a 10-team pool with 2
    centres each, 20 centres get drafted. If the 20th-best centre still scores
    140, then a 150-point centre is only worth 10 points *more than what you
    could have had anyway*. His edge is 10, not 150.

    Meanwhile if only 10 goalies get drafted and the 10th scores 60, a 120-point
    goalie has an edge of 60 -- and should go first, despite the smaller number.

    So: edge = projection - (the projection of the last player at that position
    who still gets drafted). Draft by edge, not by projection.

    Known elsewhere as VORP, value over replacement player.
    """
    df = df.copy()
    df["position_group"] = ""
    for group, positions in POSITION_GROUPS.items():
        df.loc[df["positionCode"].isin(positions), "position_group"] = group

    df["replacement"] = 0.0
    for pos, n in slots.items():
        mask = df["position_group"] == pos
        pool = df.loc[mask, "projection"].sort_values(ascending=False)
        cutoff = int(n * n_teams)
        level = pool.iloc[cutoff] if len(pool) > cutoff else (pool.min() if len(pool) else 0.0)
        df.loc[mask, "replacement"] = level
    df["edge"] = df["projection"] - df["replacement"]
    return df


# ─── App ───────────────────────────────────────────────────────────────────────

st.title(f"🏒 Fantasy Hockey — {TARGET_SEASON[:4]}-{TARGET_SEASON[4:]}")

tab_draft, tab_lab, tab_season = st.tabs(
    ["🎯 Draft Board", "🔬 Model Lab", "📈 Season Tracker"]
)


# ─── Draft board ───────────────────────────────────────────────────────────────

with tab_draft:
    st.sidebar.header("Draft settings")
    n_teams = st.sidebar.number_input("Teams in pool", 4, 20, 7)
    st.sidebar.caption("Roster slots per team, used for replacement level:")
    slots = {
        "A": st.sidebar.number_input("A", 0, 12, 12),
        "D": st.sidebar.number_input("D", 0, 8, 6),
        "G": st.sidebar.number_input("G", 0, 4, 2),
    }

    proj = add_edge(get_projections(TARGET), slots, n_teams)

    st.info(
        "**Proj** — fantasy points we expect over a full 82-game season.  \n"
        "**Edge** — how many points better than the *worst starter at that "
        "position*. In a 10-team pool with 2 centres each, only 20 centres get "
        "drafted; being better than the 20th is what actually wins you points. "
        "**Draft by Edge, not by Proj.**  \n"
        "**Last yr** — what they actually scored last season, for a sanity check.",
        icon="ℹ️",
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        search = st.text_input("Search player", "")
    with c2:
        positions = st.multiselect(
            "Positions", ["A", "D", "G"], default=["A", "D", "G"]
        )
    with c3:
        rank_by = st.selectbox("Rank by", ["edge", "projection"])

    view = proj[proj.position_group.isin(positions)]
    if search:
        view = view[view.playerName.str.contains(search, case=False, na=False)]
    view = view.sort_values(rank_by, ascending=False).copy()
    view["drafted"] = view.playerId.isin(load_roster())

    st.dataframe(
        view[["playerName", "position_group", "age", "projection", "edge",
              "prev1_fp_per82", "drafted"]]
        .rename(columns={
            "playerName": "Player", "position_group": "Pos", "age": "Age",
            "projection": "Proj", "edge": "Edge",
            "prev1_fp_per82": "Last yr", "drafted": "Mine",
        })
        .round(1)
        .reset_index(drop=True),
        use_container_width=True, hide_index=True, height=460,
    )

    st.subheader("Positional scarcity")
    st.caption(
        "Where the drop-off is steepest is where you draft early. Defencemen "
        "score 3/2 against forwards' 2/1, so the D curve sitting high is real."
    )
    curve = (
        proj.sort_values("projection", ascending=False)
        .groupby("position_group")
        .head(40)
        .assign(rank=lambda d: d.groupby("position_group").cumcount() + 1)
    )
    st.plotly_chart(
        px.line(curve, x="rank", y="projection", color="position_group", markers=True,
                labels={"rank": "Rank within position",
                "projection": "Projected fantasy pts / 82",
                "position_group": "Position"}),
        use_container_width=True,
    )

    with st.expander("My roster"):
        picks = st.multiselect(
            "Drafted players",
            options=proj.playerName.dropna().tolist(),
            default=proj[proj.playerId.isin(load_roster())].playerName.dropna().tolist(),
        )
        if st.button("Save roster"):
            save_roster(proj[proj.playerName.isin(picks)].playerId.tolist())
            st.success(f"Saved {len(picks)} players.")
            st.rerun()
        if picks:
            mine = proj[proj.playerName.isin(picks)]
            a, b, c = st.columns(3)
            a.metric("Players", len(mine))
            b.metric("Projected pts/82", f"{mine.projection.sum():.0f}")
            c.metric("Total Edge", f"{mine.edge.sum():.0f}")


# ─── Model lab ─────────────────────────────────────────────────────────────────

with tab_lab:
    st.subheader("Leaderboard")
    st.markdown(
        "Compare approaches on the same test: train on past seasons, predict the "
        "next one, repeat. **`pct_of_perfect@50` is the column that matters** — "
        "draft this model's top 50 and that is the share of a perfect draft's "
        "points you end up with."
    )

    with st.expander("What do these columns mean?"):
        st.markdown(
            """
| Column | Meaning |
|---|---|
| **pct_of_perfect@50** | Draft its top 50 → what share of a perfect draft's points you get. **The one that matters.** |
| **points_lost@50** | The same gap, in fantasy points instead of a percentage. |
| **rank_corr** | Did it put players in the right order? 1.0 perfect, 0 random. |
| **top50_hit** | Of its top 50 picks, what fraction were truly top 50. |
| **RMSE** | Typical error size in fantasy points. Lower is better — but see below. |
| **MAE** | Same idea, less harsh on big misses. |
| **bias** | Over-predicting (+) or under-predicting (−) on average. |

**Watch out:** RMSE and drafting well are not the same thing. RMSE asks whether
the numbers land close; a draft only cares about getting the *order* right. The
GBM below beats the weighted average on RMSE and still drafts no better. When
they disagree, believe `pct_of_perfect@50`.

**The rows:** *positional_mean* predicts every defenceman as the average
defenceman — the floor. *last_season* just repeats last year. *weighted_average*
blends the last 3 seasons, recent years counting more. *gbm* is gradient
boosting on all the features.
            """
        )

    lab_target = st.radio(
        "What are we predicting?", ["fp_per82", "fantasy_points"], horizontal=True,
        help=(
            "fp_per82 = fantasy points scaled to a full 82-game season. Measures "
            "how good a player is per game, and is fair across the short COVID "
            "seasons. fantasy_points = the raw season total, which is what the "
            "pool actually pays — it also punishes players who miss games."
        ),
    )

    if st.button("Run walk-forward evaluation"):
        hist = get_history()
        feats = datasets.feature_columns(hist, task="draft")
        with st.spinner("Training across seasons…"):
            runs = {
                "positional_mean": models.make_baseline("positional_mean"),
                "last_season": models.make_baseline("last_season"),
                "weighted_average": models.make_baseline("weighted_average"),
                "gbm": models.make_gbm(),
            }
            st.session_state["results"] = {
                name: evaluate.walk_forward(hist, feats, lab_target, fn)
                for name, fn in runs.items()
            }

    if "results" in st.session_state:
        st.dataframe(
            evaluate.compare(st.session_state["results"]).round(3),
            use_container_width=True,
        )
        pick = st.selectbox("Season detail", list(st.session_state["results"]))
        st.dataframe(
            st.session_state["results"][pick].round(3),
            use_container_width=True, hide_index=True,
        )

    with st.expander("Feature importance"):
        if st.button("Compute"):
            hist = get_history()
            feats = datasets.feature_columns(hist, task="draft")
            imp = models.feature_importance(hist, feats, lab_target)
            st.plotly_chart(
                px.bar(imp.head(20), x="importance", y="feature", orientation="h"),
                use_container_width=True,
            )


# ─── Season tracker ────────────────────────────────────────────────────────────

with tab_season:
    st.subheader("Season tracker")

    @st.cache_data(show_spinner="Checking for live games…", ttl=3600)
    def get_live() -> pd.DataFrame:
        try:
            return data.game_logs(TARGET_SEASON)
        except Exception:
            return pd.DataFrame()

    live = get_live()

    if live.empty:
        st.info(
            f"No {TARGET_SEASON[:4]}-{TARGET_SEASON[4:]} games played yet. "
            "This tab activates on opening night."
        )
        st.markdown(
            """
**To build** — the in-season half of the project:

1. **Refresh live data.** `data.game_logs(TARGET_SEASON)` re-fetches; delete
   `data/raw/games_20262027.parquet` to force it. Wire that to a button.
2. **Projected finish per player.** Use `datasets.inseason_table()` and predict
   `fantasy_remaining`, then add points already banked. Straight-line pace
   extrapolation is the baseline to beat, and it is strong after December.
3. **Add/drop advisor.** Compare each rostered player's projected remaining
   points against the best free agent at the same position.
4. **Schedule awareness.** `client.schedule.team_season_schedule` gives games
   per week. Four games this week beats an equal player with two.
5. **Track calibration.** Plot predicted against actual as the season runs, so
   drift shows up in November rather than April.
            """
        )
    else:
        st.success(f"{len(live):,} game rows loaded for {TARGET_SEASON}.")
        st.dataframe(live.head(50), use_container_width=True)
        st.caption("Projections not wired up yet — see step 2 above.")
