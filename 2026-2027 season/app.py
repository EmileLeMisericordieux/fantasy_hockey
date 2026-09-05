"""
Fantasy hockey dashboard, 2026-27.

    python -m streamlit run app.py

Use `python -m streamlit`, not bare `streamlit`. With several Python installs on
one machine, bare `streamlit` runs whichever PATH finds first, which may not be
the one holding these packages.

Three tabs:
  Draft Board    A working first pass. Improving it is the point.
  Model Lab      Compare approaches on the same walk-forward split.
  Season Tracker The in-season model, driven by a date slider. Works before
                 opening night by replaying a finished season.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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

# Chart palette. Slots 1-3 of the reference categorical set, used unchanged
# because that subset is the one validated for scatter plots, where every pair
# of series sits next to every other. Three series is also its documented cap
# @D@ a fourth would put yellow beside orange and fail. Aqua falls under 3:1 on
# a light surface, so anything drawn in it also appears in the table below.
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"
INK_MUTED = "#8a8a85"
POS_COLOURS = {"Forward": SERIES_BLUE, "Defence": SERIES_ORANGE, "Goalie": SERIES_AQUA}


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


@st.cache_data(show_spinner="Loading the in-season table…")
def get_inseason() -> pd.DataFrame:
    """
    Game-level table, trimmed and downcast.

    Half a million rows by 320 features is over a gigabyte in float64, which is
    more than a dashboard should hold. float32 halves it and costs nothing —
    these are rolling averages, not currency.
    """
    df = datasets.inseason_table()
    feats = datasets.feature_columns(df, task="inseason")
    keep = sorted(set(feats) | {
        "playerId", "playerName", "positionCode", "season", "season_start",
        "gameDate", "games_so_far", "team_games_left", "teamAbbrev",
        "fantasy_per_game_so_far", "prev1_fp_per_game", "cum_fantasy_so_far",
        "fantasy_remaining",
    })
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in feats:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


@st.cache_data(show_spinner="Training on every season before it…")
def get_season_predictions(season_start: int) -> pd.DataFrame:
    """
    Replay one finished season: train on everything before it, then predict
    every player-game in it.

    The honest thing about this is what the model is *not* given — no part of
    the season being replayed is in its training data, so what you see on the
    slider is what it would have said at the time.
    """
    df = get_inseason()
    feats = datasets.feature_columns(df, task="inseason")
    train = df[df.season_start < season_start]
    test = df[df.season_start == season_start].copy()

    test["pred"] = models.make_inseason_gbm(sample_every=10)(
        train, test, feats, "fantasy_remaining")
    test["baseline"] = models.make_inseason_baseline("blend")(
        train, test, feats, "fantasy_remaining")
    return test


def position_group(code: str) -> str:
    if code in GOALIE_POS:
        return "Goalie"
    if code in DEFENCE_POS:
        return "Defence"
    return "Forward"


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
        width='stretch', hide_index=True, height=460,
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
        width='stretch',
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
            width='stretch',
        )
        pick = st.selectbox("Season detail", list(st.session_state["results"]))
        st.dataframe(
            st.session_state["results"][pick].round(3),
            width='stretch', hide_index=True,
        )

    with st.expander("Feature importance"):
        if st.button("Compute"):
            hist = get_history()
            feats = datasets.feature_columns(hist, task="draft")
            imp = models.feature_importance(hist, feats, lab_target)
            st.plotly_chart(
                px.bar(imp.head(20), x="importance", y="feature", orientation="h"),
                width='stretch',
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
    if not live.empty:
        st.success(f"{len(live):,} game rows loaded for {TARGET_SEASON}.")

    st.markdown(
        f"""
### Time machine

{TARGET_SEASON[:4]}-{TARGET_SEASON[4:]} has not started, so there is nothing
live to project. Instead, **stand at a date inside a finished season** and see
what the in-season model would have told you that morning.

It is trained only on seasons *before* the one you are replaying, so nothing on
screen is hindsight. Move the slider and watch the model get less useful as the
season runs out — that decay is the real finding here.
        """
    )

    inseason = get_inseason()
    seasons = sorted(inseason["season_start"].unique())
    replayable = [int(x) for x in seasons if x >= seasons[0] + 3]

    c1, c2 = st.columns([1, 3])
    with c1:
        season_start = st.selectbox(
            "Replay season", replayable, index=len(replayable) - 1,
            format_func=lambda y: f"{y}-{str(y + 1)[2:]}",
        )
        metric_label = st.radio(
            "Track", ["% of perfect @30", "Rank correlation", "Top-25 hit rate"],
            help=(
                "% of perfect: draft the model's top 30 for the rest of the "
                "season and you collect this share of what the best possible 30 "
                "would have scored. The one that matters."
            ),
        )

    preds = get_season_predictions(int(season_start))
    day_one, last_day = preds["gameDate"].min(), preds["gameDate"].max()

    with c2:
        as_of = st.slider(
            "You are standing on",
            min_value=day_one.to_pydatetime(), max_value=last_day.to_pydatetime(),
            value=(day_one + (last_day - day_one) * 0.25).to_pydatetime(),
            format="D MMM YYYY",
        )
    as_of = pd.Timestamp(as_of)

    METRIC_KEY = {
        "% of perfect @30": ("pct_of_perfect@30", "% of a perfect 30"),
        "Rank correlation": ("rank_corr", "Rank correlation"),
        "Top-25 hit rate": ("top25_hit", "Top-25 hit rate"),
    }
    key, axis_title = METRIC_KEY[metric_label]

    def score(field: pd.DataFrame, col: str) -> dict:
        return evaluate.all_metrics(
            field["fantasy_remaining"].to_numpy(), field[col].to_numpy(), n_picks=30
        )

    field = evaluate.snapshot(preds, as_of)
    field = field[field["games_so_far"] >= 1]

    if len(field) < 30:
        st.warning("Too few players have played by this date. Move the slider on.")
    else:
        m_model, m_base = score(field, "pred"), score(field, "baseline")

        st.markdown(f"#### The field on {as_of.day} {as_of:%B %Y}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("% of a perfect 30", f"{m_model['pct_of_perfect@30']:.1%}",
                  f"{m_model['pct_of_perfect@30'] - m_base['pct_of_perfect@30']:+.1%} vs baseline")
        k2.metric("Rank correlation", f"{m_model['rank_corr']:.3f}",
                  f"{m_model['rank_corr'] - m_base['rank_corr']:+.3f} vs baseline")
        k3.metric("Top-25 hit rate", f"{m_model['top25_hit']:.1%}",
                  f"{m_model['top25_hit'] - m_base['top25_hit']:+.1%} vs baseline")
        k4.metric("Players available", f"{len(field):,}",
                  f"{field['team_games_left'].median():.0f} games left")

        # ── How the model holds up across the whole season ────────────────────
        @st.cache_data(show_spinner="Scoring every week of the season…")
        def season_curve(season_start: int) -> pd.DataFrame:
            p = get_season_predictions(season_start)
            d0, d1 = p["gameDate"].min(), p["gameDate"].max()
            rows = []
            for frac in np.linspace(0.04, 0.96, 22):
                when = d0 + (d1 - d0) * frac
                f = evaluate.snapshot(p, when)
                f = f[f["games_so_far"] >= 1]
                if len(f) < 30:
                    continue
                for name, col in (("Model", "pred"), ("Baseline (blend)", "baseline")):
                    rows.append({"date": when, "series": name,
                                 **evaluate.all_metrics(
                                     f["fantasy_remaining"].to_numpy(),
                                     f[col].to_numpy(), n_picks=30)})
            return pd.DataFrame(rows)

        curve = season_curve(int(season_start))

        st.markdown(f"#### {axis_title}, across the season")
        fig = px.line(
            curve, x="date", y=key, color="series",
            color_discrete_map={"Model": SERIES_BLUE, "Baseline (blend)": SERIES_ORANGE},
            labels={"date": "", key: axis_title, "series": ""},
        )
        fig.update_traces(line=dict(width=2), hovertemplate="%{y:.3f}<extra></extra>")
        fig.add_vline(x=as_of, line_width=1, line_dash="dot", line_color=INK_MUTED,
                      annotation_text="you are here", annotation_position="top",
                      annotation_font_color=INK_MUTED)
        fig.update_layout(
            hovermode="x unified", height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(showgrid=False, linecolor=INK_MUTED)
        fig.update_yaxes(gridcolor="rgba(138,138,133,0.25)", zeroline=False)
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Both lines fall as the season runs out: with twenty games left, "
            "who scores most is more luck than skill, and no model fixes that. "
            "The gap between them is what the model is worth, and it is widest "
            "in autumn — when pace is a dozen games of noise and last season "
            "is the only real evidence."
        )

        # ── What it said, against what happened ──────────────────────────────
        st.markdown("#### Predicted against actual, on this date")
        plot = field.assign(
            Position=field["positionCode"].map(position_group),
            Player=field["playerName"],
        )
        lim = float(max(plot["pred"].max(), plot["fantasy_remaining"].max())) * 1.05
        sc = px.scatter(
            plot, x="pred", y="fantasy_remaining", color="Position",
            color_discrete_map=POS_COLOURS, hover_name="Player",
            hover_data={"pred": ":.1f", "fantasy_remaining": ":.1f", "Position": False},
            labels={"pred": "Predicted points from here",
                    "fantasy_remaining": "What he actually scored"},
        )
        sc.update_traces(marker=dict(size=8, opacity=0.65,
                                     line=dict(width=1, color="rgba(255,255,255,0.85)")))
        sc.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                     line=dict(color=INK_MUTED, width=1, dash="dot"))
        sc.add_annotation(x=lim * 0.82, y=lim * 0.88, text="perfect prediction",
                          showarrow=False, font=dict(color=INK_MUTED, size=11))
        sc.update_layout(
            height=440, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        sc.update_xaxes(gridcolor="rgba(138,138,133,0.25)", zeroline=False)
        sc.update_yaxes(gridcolor="rgba(138,138,133,0.25)", zeroline=False)
        st.plotly_chart(sc, width='stretch')
        st.caption(
            "Points above the dotted line outscored the projection; below it, "
            "they fell short. The cloud is wide on purpose — this is an "
            "honest picture of how much of a hockey season is noise."
        )

        # ── The picks themselves ─────────────────────────────────────────────
        st.markdown("#### Who it would have told you to hold")
        top = field.nlargest(30, "pred").copy()
        actual_rank = field["fantasy_remaining"].rank(ascending=False, method="min")
        top["Actual rank"] = actual_rank.reindex(top.index).astype(int)
        top["Position"] = top["positionCode"].map(position_group)
        table = pd.DataFrame({
            "Player": top["playerName"].values,
            "Pos": top["Position"].values,
            "Team": top["teamAbbrev"].values,
            "Predicted": top["pred"].round(1).values,
            "Actual": top["fantasy_remaining"].round(1).values,
            "Actual rank": top["Actual rank"].values,
            "Banked so far": top["cum_fantasy_so_far"].round(0).values,
        })
        st.dataframe(table, width='stretch', hide_index=True)
        hits = int((top["Actual rank"] <= 30).sum())
        st.caption(
            f"{hits} of these 30 really were the 30 best from this date. "
            "The table is also the accessible reading of the charts above — "
            "every number in them is here in text."
        )

    with st.expander("Still to build"):
        st.markdown(
            """
1. **Live data.** `data.game_logs(TARGET_SEASON)` re-fetches; delete
   `data/raw/games_20262027.parquet` to force it. Wire that to a button.
2. **Add/drop advisor.** Compare each rostered player's projected remaining
   points against the best free agent at the same position. The model is
   already producing the number; nothing consumes it.
3. **Schedule awareness for the week ahead.** `team_games_left` is a
   season-long count. Four games *this week* beats an equal player with two.
4. **Injuries.** Nothing here knows a player is hurt; it infers it from ice
   time drying up, several games late.
            """
        )
