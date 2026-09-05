"""
Fantasy Hockey Dashboard
Run with:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
from xgboost import XGBRegressor

import re

from feature_engineering import FantasyDataEngineer

# ── Config ─────────────────────────────────────────────────────────────────────

def _current_nhl_season() -> tuple[int, int]:
    """Return (start_year, end_year) of the current NHL season based on today's date.
    NHL seasons run Oct–Apr: Oct-Dec of year Y → Y/{Y+1}, Jan-Sep of year Y → {Y-1}/Y.
    During the off-season (May–Sep) we return the upcoming season.
    """
    today = pd.Timestamp.today()
    if today.month >= 10:
        return today.year, today.year + 1
    else:
        return today.year - 1, today.year


_CUR_START, _CUR_END = _current_nhl_season()
_CURRENT_SEASON_LABEL = f"{_CUR_START}-{_CUR_END}"  # e.g. "2025-2026"

TRAIN_SEASONS   = [f"{y}{y+1}" for y in range(2020, _CUR_START)]
SCORE_SEASONS   = [f"{_CUR_START - 1}{_CUR_START}"]
CURRENT_SEASONS = [f"{_CUR_START}{_CUR_END}"]

MIN_GAMES = 5

RAW_STAT_COLS = [
    "goals", "assists", "points", "shots", "toi", "timeOnIce", "toi_minutes",
    "powerPlayGoals", "powerPlayAssists", "shortHandedGoals", "shortHandedAssists",
    "plusMinus", "pim", "decision", "homeRoadFlag", "fantasy_points", "p60",
]
META_COLS = [
    "playerId", "seasonId", "gameId", "season", "season_start_year", "gameDate",
    "skaterFullName", "lastName", "teamAbbrevs", "opponentTeamAbbrev",
]
TARGET = "season_fantasy_remaining"
TOTAL_COL = "season_fantasy_total"
CAT_COLS_CANDIDATES = ["positionCode", "shootsCatches"]

CURRENT_SEASONS = ["20252026"]

_STAT_LABELS = {
    "fantasy_points": "fantasy pts",
    "goals":          "goals",
    "assists":        "assists",
    "shots":          "shots on goal",
    "toi_minutes":    "ice time (min)",
    "plusMinus":      "+/−",
    "p60":            "pts per 60 min",
}

_FEATURE_LABELS = {
    "cum_fantasy_so_far":  "cumulative fantasy pts (before this game)",
    "games_remaining":     "games remaining in season",
    "season_progress":     "season completion (%)",
    "projected_pace":      "projected season pace",
    "points_needed_pace":  "pts needed to match pace",
    "days_rest":           "days of rest before game",
    "age":                 "player age",
    "games_in_season":     "games played this season",
    "overall_pick":        "draft pick position",
    "rookie_year":         "years since debut",
    "height":              "height",
    "weight":              "weight",
}

def prettify_feature(name: str) -> str:
    m = re.match(r"roll(\d+)_(.+)", name)
    if m:
        n, stat = m.group(1), m.group(2)
        return f"{n}-game avg {_STAT_LABELS.get(stat, stat.replace('_', ' '))}"
    return _FEATURE_LABELS.get(name, name.replace("_", " "))


ROSTER_FILE   = Path("roster.json")

def load_roster() -> list[str]:
    if ROSTER_FILE.exists():
        import json
        return json.loads(ROSTER_FILE.read_text())
    return []

def save_roster(roster: list[str]):
    import json
    ROSTER_FILE.write_text(json.dumps(roster))


TRAIN_CACHE   = Path("cache_game_df_train.parquet")
SCORE_CACHE   = Path("cache_game_df_score.parquet")
CURRENT_CACHE = Path("cache_game_df_current.parquet")


# ── Data loading (cached) ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training model…")
def load_model():
    if TRAIN_CACHE.exists():
        train_df = pd.read_parquet(TRAIN_CACHE)
        if TARGET not in train_df.columns:
            train_df = None
    else:
        train_df = None

    if train_df is None:
        eng = FantasyDataEngineer(seasons=TRAIN_SEASONS, limit=100, debug=False)
        train_df = eng.build_game_dataset()
        train_df.to_parquet(TRAIN_CACHE, index=False)

    df = train_df[train_df["games_in_season"] >= MIN_GAMES].copy()
    cat_cols = [c for c in CAT_COLS_CANDIDATES if c in df.columns]
    df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    drop_cols = set(RAW_STAT_COLS + META_COLS + [TARGET, TOTAL_COL]) & set(df_model.columns)
    feature_cols = [
        c for c in df_model.columns
        if c not in drop_cols and df_model[c].dtype != object
    ]

    X = df_model[feature_cols].dropna()
    y = df_model.loc[X.index, TARGET]

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:quantileerror", quantile_alpha=0.65, random_state=42, tree_method="hist",
    )
    model.fit(X, y)
    return model, feature_cols


@st.cache_data(show_spinner="Fetching 2024-25 season data…")
def load_score_df():
    if SCORE_CACHE.exists():
        return pd.read_parquet(SCORE_CACHE)
    eng = FantasyDataEngineer(seasons=SCORE_SEASONS, limit=100, debug=False)
    df = eng.build_game_dataset()
    df.to_parquet(SCORE_CACHE, index=False)
    return df


@st.cache_data(show_spinner=f"Fetching {_CURRENT_SEASON_LABEL} season data…")
def load_current_df():
    if CURRENT_CACHE.exists():
        return pd.read_parquet(CURRENT_CACHE)
    eng = FantasyDataEngineer(seasons=CURRENT_SEASONS, limit=100, debug=False)
    df = eng.build_game_dataset()
    df.to_parquet(CURRENT_CACHE, index=False)
    return df


def predict_current_season(score_df, model, feature_cols):
    """
    For each player, predict their end-of-season total from their most recent
    game row (which encodes current rolling form).
    """
    df = score_df[score_df["games_in_season"] >= MIN_GAMES].copy()
    cat_cols = [c for c in CAT_COLS_CANDIDATES if c in df.columns]

    # preserve columns that get_dummies will drop
    passthrough = [c for c in ["positionCode", "cum_fantasy_so_far", "fantasy_points"] if c in df.columns]
    passthrough_vals = df[passthrough].copy()

    df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    latest = (
        df_model.sort_values("gameDate")
        .groupby("playerId", as_index=False)
        .last()
    )
    latest_passthrough = (
        passthrough_vals.loc[df_model.sort_values("gameDate").groupby("playerId").tail(1).index]
        .reset_index(drop=True)
    )

    X = latest.reindex(columns=feature_cols, fill_value=0)
    latest = latest.copy().reset_index(drop=True)
    for col in passthrough:
        latest[col] = latest_passthrough[col].values

    # true earned so far = cum_fantasy_so_far (shift-1) + current game's points
    if "cum_fantasy_so_far" in latest.columns and "fantasy_points" in latest.columns:
        latest["earned_so_far"] = latest["cum_fantasy_so_far"] + latest["fantasy_points"].fillna(0)
    elif "cum_fantasy_so_far" in latest.columns:
        latest["earned_so_far"] = latest["cum_fantasy_so_far"]
    else:
        latest["earned_so_far"] = 0

    # model predicts remaining points after current game; add back what's already earned
    predicted_remaining = model.predict(X).clip(min=0)
    latest["predicted_remaining"] = predicted_remaining
    latest["predicted_season_total"] = latest["earned_so_far"] + predicted_remaining

    return latest


def build_time_series(score_df, model, feature_cols, player_ids):
    """
    For selected players: at each game date, use the rolling features as of
    that game to predict the season total — giving a trajectory of how the
    model's forecast evolved game by game.
    """
    df = score_df[score_df["playerId"].isin(player_ids)].copy()
    df = df[df["games_in_season"] >= MIN_GAMES].copy()
    cat_cols = [c for c in CAT_COLS_CANDIDATES if c in df.columns]
    # preserve positionCode before get_dummies consumes it
    position_col = df["positionCode"].values if "positionCode" in df.columns else None
    df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df_model.reindex(columns=feature_cols, fill_value=0)
    df_model = df_model.copy()
    df_model["predicted_season_total"] = model.predict(X)
    if position_col is not None:
        df_model["positionCode"] = position_col
    return_cols = ["playerId", "skaterFullName", "gameDate", "predicted_season_total"] + \
                  (["positionCode"] if position_col is not None else [])
    return df_model[return_cols]


# ── App ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fantasy Hockey", layout="wide")
st.title("Fantasy Hockey — Season Predictions")

with st.spinner("Loading…"):
    model, feature_cols = load_model()
    score_df = load_score_df()

score_df["gameDate"] = pd.to_datetime(score_df["gameDate"])

# ── compute_metrics (module-level so @st.cache_data works outside a with block) ─

@st.cache_data(show_spinner="Computing walk-forward metrics…")
def compute_metrics(_model, _feature_cols, train_cache_path):
    train_df = pd.read_parquet(train_cache_path)
    df = train_df[train_df["games_in_season"] >= MIN_GAMES].copy()
    cat_cols = [c for c in CAT_COLS_CANDIDATES if c in df.columns]
    df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    drop_cols = set(RAW_STAT_COLS + META_COLS + [TARGET, TOTAL_COL]) & set(df_model.columns)
    feature_cols_local = [c for c in df_model.columns if c not in drop_cols and df_model[c].dtype != object]

    unique_seasons = sorted(df_model["season_start_year"].unique())
    rows = []
    for cutoff in unique_seasons[:-1]:
        train = df_model[df_model["season_start_year"] <= cutoff].dropna(subset=[TARGET])
        test  = df_model[df_model["season_start_year"] == cutoff + 1].dropna(subset=[TARGET])
        if len(test) == 0:
            continue
        X_train = train[feature_cols_local]
        X_test  = test[feature_cols_local].reindex(columns=feature_cols_local, fill_value=0)
        m = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:quantileerror", quantile_alpha=0.65,
            random_state=42, tree_method="hist",
        )
        m.fit(X_train, train[TARGET])
        pred_remaining = m.predict(X_test).clip(min=0)
        pred_total = test["cum_fantasy_so_far"].values + pred_remaining
        actual_total = test[TOTAL_COL].values
        errors = pred_total - actual_total
        rows.append({
            "Season":      f"{cutoff}-{cutoff+1}",
            "Test rows":   len(test),
            "RMSE":        round(float(((errors**2).mean())**0.5), 2),
            "MAE":         round(float(abs(errors).mean()), 2),
            "Bias":        round(float(errors.mean()), 2),
            "Within ±10": f"{(abs(errors) <= 10).mean()*100:.1f}%",
            "Within ±20": f"{(abs(errors) <= 20).mean()*100:.1f}%",
        })
        last_test_actual = actual_total
        last_test_pred   = pred_total
        last_cutoff      = cutoff
        last_test_names  = test["skaterFullName"].values if "skaterFullName" in test.columns else None

    return pd.DataFrame(rows), last_test_actual, last_test_pred, last_test_names, last_cutoff


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_monitor, tab_live, tab_roster = st.tabs(["📊 Model Performance", "🏒 Live Predictions", "📋 My Roster"])

# ── Model performance tab ──────────────────────────────────────────────────────

with tab_monitor:
    metrics_df, last_actual, last_pred, last_names, last_cutoff = compute_metrics(
        model, feature_cols, TRAIN_CACHE
    )

    st.subheader("Walk-forward CV by season")
    st.caption("Each row trains on all prior seasons, tests on the next. RMSE/MAE are on predicted season totals.")
    st.dataframe(metrics_df.set_index("Season"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_rmse = px.line(
            metrics_df, x="Season", y="RMSE", markers=True,
            title="RMSE by Season", labels={"RMSE": "RMSE (fantasy pts)"},
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    with col2:
        fig_bias = px.bar(
            metrics_df, x="Season", y="Bias",
            title="Bias by Season (+ = overestimate, − = underestimate)",
            labels={"Bias": "Mean error (fantasy pts)"},
            color="Bias", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
        )
        st.plotly_chart(fig_bias, use_container_width=True)

    st.subheader(f"Actual vs Predicted — test season {last_cutoff+1}")
    scatter_df = pd.DataFrame({
        "actual":    last_actual,
        "predicted": last_pred,
        "player":    last_names if last_names is not None else [""] * len(last_actual),
    })
    max_val = max(scatter_df[["actual", "predicted"]].max()) * 1.05
    fig_scatter = px.scatter(
        scatter_df, x="actual", y="predicted", hover_name="player",
        labels={"actual": "Actual season total", "predicted": "Predicted season total"},
        title=f"Actual vs Predicted fantasy points ({last_cutoff+1} season)",
        opacity=0.6,
    )
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                          line=dict(dash="dash", color="grey", width=1))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Live predictions tab ───────────────────────────────────────────────────────

with tab_live:
    current_df = load_current_df()
    current_df["gameDate"] = pd.to_datetime(current_df["gameDate"])

    # ── Date slicer (sidebar) ──────────────────────────────────────────────────
    st.sidebar.header("⚙️ Settings")
    st.sidebar.subheader("Prediction Date")
    min_date = current_df["gameDate"].min().date()
    max_date = current_df["gameDate"].max().date()
    as_of_date = st.sidebar.slider(
        "Predict as of",
        min_value=min_date,
        max_value=max_date,
        value=max_date,
        format="YYYY-MM-DD",
    )
    current_df_asof = current_df[current_df["gameDate"] <= pd.Timestamp(as_of_date)]

    current_latest = predict_current_season(current_df_asof, model, feature_cols)
    current_all_players = (
        current_latest.sort_values("predicted_season_total", ascending=False)
        ["skaterFullName"].dropna().unique().tolist()
    )

    current_selected = st.multiselect(
        f"Select players ({_CURRENT_SEASON_LABEL})",
        current_all_players,
        default=current_all_players[:5],
        key="current_players",
    )
    explanation_placeholder = st.empty()

    if current_selected:
        cur_ids = current_latest[current_latest["skaterFullName"].isin(current_selected)]["playerId"].tolist()

        # One row per player-game with true cumulative fantasy points
        actual_df = (
            current_df_asof[current_df_asof["playerId"].isin(cur_ids)]
            [["playerId", "skaterFullName", "gameDate", "fantasy_points"]]
            .dropna(subset=["fantasy_points"])
            .sort_values(["skaterFullName", "gameDate"])
            .reset_index(drop=True)
        )
        actual_df["true_cumulative"] = actual_df.groupby("playerId")["fantasy_points"].cumsum()

        pred_map = (
            current_latest[current_latest["playerId"].isin(cur_ids)]
            .set_index("skaterFullName")["predicted_season_total"]
            .to_dict()
        )
        remaining_map = (
            current_latest[current_latest["playerId"].isin(cur_ids)]
            .set_index("skaterFullName")["predicted_remaining"].to_dict()
            if "predicted_remaining" in current_latest.columns else {}
        )
        cum_map = (
            actual_df.sort_values("gameDate")
            .groupby("skaterFullName")["true_cumulative"]
            .last()
            .to_dict()
        )

        SEASON_LENGTH   = 82
        SEASON_END_DATE = pd.Timestamp(f"{_CUR_END}-04-18")

        # Games remaining in the season from today (same for all players)
        _season_start       = current_df["gameDate"].min()
        _total_season_days  = max((SEASON_END_DATE - _season_start).days, 1)
        _days_remaining     = max((SEASON_END_DATE - pd.Timestamp.today().normalize()).days, 0)
        GAMES_REMAINING_NOW = round(SEASON_LENGTH * _days_remaining / _total_season_days)

        def hex_to_rgba(hex_colour: str, alpha: float = 0.15) -> str:
            h = hex_colour.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        palette = px.colors.qualitative.Plotly
        fig_cur = go.Figure()

        for i, name in enumerate(current_selected):
            if name not in remaining_map:
                continue
            colour          = palette[i % len(palette)]
            predicted_total = round(cum_map.get(name, 0) + remaining_map.get(name, 0), 2)
            player_actual   = actual_df[actual_df["skaterFullName"] == name].reset_index(drop=True)

            if player_actual.empty:
                continue

            today        = pd.Timestamp.today().normalize()
            last_date    = player_actual["gameDate"].max()
            games_played = len(player_actual)
            games_left   = GAMES_REMAINING_NOW
            pts_per_game = predicted_total / SEASON_LENGTH

            actual_y = [
                round(max(pts_per_game * (k + 1), float(player_actual["true_cumulative"].iloc[k])), 2)
                for k in range(games_played)
            ]

            # extend the actual series to today with the player's latest cumulative score
            earned_today = float(player_actual["true_cumulative"].iloc[-1])
            solid_dates  = list(player_actual["gameDate"]) + ([today] if today > last_date else [])
            solid_y      = actual_y + ([round(max(actual_y[-1], earned_today), 2)] if today > last_date else [])

            # shaded area: true cumulative extended to today
            shade_dates = list(player_actual["gameDate"]) + ([today] if today > last_date else [])
            shade_y     = list(player_actual["true_cumulative"]) + ([earned_today] if today > last_date else [])

            fig_cur.add_trace(go.Scatter(
                x=shade_dates,
                y=shade_y,
                mode="none",
                name=f"{name} (earned so far)",
                legendgroup=name,
                fill="tozeroy",
                fillcolor=hex_to_rgba(colour, alpha=0.15),
                showlegend=True,
            ))
            fig_cur.add_trace(go.Scatter(
                x=solid_dates,
                y=solid_y,
                mode="lines+markers",
                name=name,
                legendgroup=name,
                line=dict(color=colour, width=2),
                marker=dict(size=4),
                showlegend=True,
            ))

            if games_left > 0:
                proj_start   = solid_y[-1]
                proj_end     = max(predicted_total, proj_start)
                anchor_date  = solid_dates[-1]
                total_days   = max((SEASON_END_DATE - anchor_date).days, games_left)
                future_dates = [
                    anchor_date + pd.Timedelta(days=int(total_days * k / games_left))
                    for k in range(0, games_left + 1)
                ]
                future_y = [
                    round(proj_start + (proj_end - proj_start) * k / games_left, 2)
                    for k in range(0, games_left + 1)
                ]
                fig_cur.add_trace(go.Scatter(
                    x=future_dates, y=future_y,
                    mode="lines",
                    name=f"{name} (projection)",
                    legendgroup=name,
                    line=dict(color=colour, width=2, dash="dash"),
                    showlegend=True,
                ))
                fig_cur.add_trace(go.Scatter(
                    x=[future_dates[-1]], y=[proj_end],
                    mode="markers",
                    legendgroup=name,
                    marker=dict(color=colour, size=9, symbol="diamond"),
                    showlegend=False,
                ))

        fig_cur.update_layout(
            title=f"{_CURRENT_SEASON_LABEL} Predicted Season Total — Running Trajectory",
            xaxis_title="Game Date",
            yaxis_title="Fantasy Points (cumulative)",
            legend_title_text="Player",
        )
        fig_cur.add_vline(
            x=pd.Timestamp.today().normalize().timestamp() * 1000,
            line_dash="dot", line_color="grey", line_width=1,
            annotation_text="today", annotation_position="top right",
        )
        st.plotly_chart(fig_cur, use_container_width=True)

        cat_cols_used = [c for c in CAT_COLS_CANDIDATES if c in current_df_asof.columns]
        features_df = (
            current_df_asof[current_df_asof["playerId"].isin(cur_ids)]
            .sort_values("gameDate")
            .groupby("playerId", as_index=False)
            .last()
        )
        features_df = pd.get_dummies(features_df, columns=cat_cols_used, drop_first=True)

        # ── SHAP feature contributions ──────────────────────────────────────────
        st.subheader("SHAP Feature Contributions")
        _valid = [n for n in current_selected if n in features_df["skaterFullName"].values]
        shap_player = _valid[0] if _valid else None
        if shap_player and len(current_selected) > 1:
            st.caption(f"Showing explanation for **{shap_player}** (first selected player).")

        if shap_player:
            player_X = (
                features_df[features_df["skaterFullName"] == shap_player]
                .reindex(columns=feature_cols, fill_value=0)
                .iloc[[0]]
            )
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(player_X)

            TOP_N      = 15
            base_value = float(explainer.expected_value)
            shap_arr   = shap_vals[0]

            ranked = sorted(zip(feature_cols, shap_arr), key=lambda t: abs(t[1]), reverse=True)
            top    = ranked[:TOP_N]
            rest   = ranked[TOP_N:]

            # ── natural-language explanation ────────────────────────────────
            earned_txt    = cum_map.get(shap_player, 0)
            remaining_txt = remaining_map.get(shap_player, 0)
            total_txt     = earned_txt + remaining_txt
            gp_txt        = len(actual_df[actual_df["skaterFullName"] == shap_player])
            gl_txt        = GAMES_REMAINING_NOW
            feat_vals     = player_X.iloc[0].to_dict()

            boosters = [(f, v) for f, v in ranked if v > 0][:3]
            drags    = [(f, v) for f, v in ranked if v < 0][:3]

            def feature_context(feat: str, val: float) -> str:
                m = re.match(r"roll(\d+)_(.+)", feat)
                if m:
                    n, stat = int(m.group(1)), m.group(2)
                    if stat == "fantasy_points":
                        return f"averaging **{val:.1f} fantasy pts/game** over the last {n} games"
                    if stat == "goals":
                        return f"scoring **{val:.2f} goals/game** over the last {n} games"
                    if stat == "assists":
                        return f"recording **{val:.2f} assists/game** over the last {n} games"
                    if stat == "shots":
                        return f"taking **{val:.1f} shots/game** over the last {n} games"
                    if stat == "toi_minutes":
                        return f"averaging **{val:.1f} min of ice time** over the last {n} games"
                    if stat == "plusMinus":
                        return f"posting a **{val:+.1f} +/−** over the last {n} games"
                    if stat == "p60":
                        return f"generating **{val:.2f} pts per 60 min** over the last {n} games"
                ctx_map = {
                    "games_remaining":      f"with **{int(val)} games remaining** in the season",
                    "season_progress":      f"being **{val*100:.0f}% through** the season",
                    "cum_fantasy_so_far":   f"having accumulated **{val:.0f} pts** before this game",
                    "projected_pace":       f"on a projected pace of **{val:.0f} pts** for the season",
                    "points_needed_pace":   f"needing **{val:.0f} more pts** to hit projected pace",
                    "days_rest":            f"coming off **{int(val)} day(s) of rest**",
                    "age":                  f"at age **{int(val)}**",
                    "games_in_season":      f"having played **{int(val)} games** this season",
                    "ctx_shootingPct":      f"with a season shooting % of **{val*100:.1f}%**",
                    "ctx_pointsPerGame":    f"averaging **{val:.2f} pts/game** on the season",
                    "ctx_powerPlayPoints":  f"with **{val:.0f} power-play pts** on the season",
                    "ctx_gamesPlayed":      f"having played **{int(val)} season games** (season summary)",
                    "ctx_faceoffWinPct":    f"winning **{val*100:.1f}%** of faceoffs this season",
                    "ctx_timeOnIcePerGame": f"averaging **{val:.1f} min of ice time/game** this season",
                    "ctx_savePct":          f"with a season save % of **{val:.3f}**",
                }
                return ctx_map.get(feat, f"**{prettify_feature(feat)}** of **{val:.2f}**")

            def feat_sentence(feat: str, shap_val: float) -> str:
                raw = feat_vals.get(feat)
                ctx = (
                    feature_context(feat, raw).capitalize()
                    if raw is not None and not pd.isna(raw)
                    else f"**{prettify_feature(feat)}**"
                )
                direction = "boosting" if shap_val > 0 else "reducing"
                return f"- {ctx}, {direction} the prediction by **{abs(shap_val):.1f} pts**"

            lines = [
                f"**{shap_player}** has earned **{earned_txt:.0f} fantasy pts** over {gp_txt} games played. "
                f"With approximately **{gl_txt} games remaining** in the season, the model predicts **{remaining_txt:.0f} more pts**, "
                f"for a projected season total of **{total_txt:.0f} pts**.",
            ]
            if boosters:
                lines.append("\n**Key factors boosting the prediction:**")
                lines += [feat_sentence(f, v) for f, v in boosters]
            if drags:
                lines.append("\n**Key factors pulling the prediction down:**")
                lines += [feat_sentence(f, v) for f, v in drags]

            explanation_placeholder.markdown("\n".join(lines))

            # ── Waterfall chart ─────────────────────────────────────────────
            wf_names    = ["Base value"]
            wf_values   = [base_value]
            wf_measures = ["absolute"]

            if rest:
                wf_names.append(f"{len(rest)} other features")
                wf_values.append(float(sum(v for _, v in rest)))
                wf_measures.append("relative")

            for feat_name, feat_val in sorted(top, key=lambda t: abs(t[1])):
                wf_names.append(feat_name)
                wf_values.append(float(feat_val))
                wf_measures.append("relative")

            prediction = base_value + float(sum(shap_arr))
            wf_names.append("Prediction")
            wf_values.append(prediction)
            wf_measures.append("total")

            fig_shap = go.Figure(go.Waterfall(
                orientation="h",
                measure=wf_measures,
                x=wf_values,
                y=wf_names,
                textposition="outside",
                text=[
                    f"{v:+.1f}" if m == "relative" else f"{v:.1f}"
                    for v, m in zip(wf_values, wf_measures)
                ],
                connector=dict(line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dot")),
                increasing=dict(marker=dict(color="#d73027", line=dict(width=0))),
                decreasing=dict(marker=dict(color="#4575b4", line=dict(width=0))),
                totals=dict(marker=dict(color="#444444", line=dict(width=0))),
                hovertemplate="%{y}<br>%{x:+.2f} pts<extra></extra>",
            ))
            fig_shap.update_layout(
                title=dict(
                    text=f"What drove the prediction for <b>{shap_player}</b>?",
                    font=dict(size=16),
                    x=0,
                ),
                xaxis=dict(
                    title="Predicted remaining fantasy pts",
                    title_font=dict(size=12, color="grey"),
                    zeroline=False,
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.06)",
                    tickfont=dict(size=11),
                ),
                yaxis=dict(title=None, tickfont=dict(size=12), showgrid=False),
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=10, r=80, t=50, b=40),
                height=max(400, 28 * (TOP_N + 4)),
                showlegend=False,
            )
            st.plotly_chart(fig_shap, use_container_width=True)

    else:
        st.info(f"Select at least one player above to see their {_CURRENT_SEASON_LABEL} trajectory.")

# ── Roster tab ─────────────────────────────────────────────────────────────────

with tab_roster:
    st.header("My Fantasy Roster")

    # Load current season data — reuse already-loaded data if the live tab ran first
    _cdf = load_current_df().copy()
    _cdf["gameDate"] = pd.to_datetime(_cdf["gameDate"])

    _roster_latest = predict_current_season(_cdf, model, feature_cols)

    # Compute true cumulative (fantasy_points.cumsum()) per player — same as live tab
    _true_cum = (
        _cdf[["playerId", "gameDate", "fantasy_points"]]
        .dropna(subset=["fantasy_points"])
        .sort_values(["playerId", "gameDate"])
        .groupby("playerId")
        .apply(lambda g: g["fantasy_points"].sum(), include_groups=False)
        .rename("true_earned")
        .reset_index()
    )
    _roster_latest = _roster_latest.merge(_true_cum, on="playerId", how="left")
    _roster_latest["true_earned"]             = _roster_latest["true_earned"].fillna(0)
    _roster_latest["predicted_season_total"]  = (_roster_latest["true_earned"] + _roster_latest["predicted_remaining"]).round(1)
    _roster_latest["earned_so_far"]           = _roster_latest["true_earned"].round(1)

    all_players_df = (
        _roster_latest[["skaterFullName", "positionCode", "predicted_season_total",
                         "predicted_remaining", "earned_so_far"]]
        .dropna(subset=["skaterFullName"])
        .copy()
    )
    all_players_df["predicted_remaining"] = all_players_df["predicted_remaining"].round(1)
    all_player_names = sorted(all_players_df["skaterFullName"].dropna().unique().tolist())

    # Persist roster in session state, backed by file
    if "roster" not in st.session_state:
        st.session_state.roster = load_roster()

    # ── Add players ───────────────────────────────────────────────────────────
    st.subheader("Manage Roster")
    col_add, col_remove = st.columns(2)
    with col_add:
        to_add = st.multiselect(
            "Add players to roster",
            [p for p in all_player_names if p not in st.session_state.roster],
            key="add_players",
        )
        if st.button("Add selected", key="btn_add") and to_add:
            st.session_state.roster = list(dict.fromkeys(st.session_state.roster + to_add))
            save_roster(st.session_state.roster)
            st.rerun()

    with col_remove:
        to_remove = st.multiselect(
            "Remove players from roster",
            st.session_state.roster,
            key="remove_players",
        )
        if st.button("Remove selected", key="btn_remove") and to_remove:
            st.session_state.roster = [p for p in st.session_state.roster if p not in to_remove]
            save_roster(st.session_state.roster)
            st.rerun()

    if not st.session_state.roster:
        st.info("Your roster is empty. Add players above.")
    else:
        # ── Roster table ──────────────────────────────────────────────────────
        st.subheader("Current Roster")
        roster_df = (
            all_players_df[all_players_df["skaterFullName"].isin(st.session_state.roster)]
            .sort_values("predicted_season_total", ascending=False)
            .rename(columns={
                "skaterFullName":         "Player",
                "positionCode":           "Pos",
                "earned_so_far":          "Earned",
                "predicted_remaining":    "Predicted Remaining",
                "predicted_season_total": "Projected Total",
            })
            .reset_index(drop=True)
        )

        total_earned    = roster_df["Earned"].sum()
        total_projected = roster_df["Projected Total"].sum()
        total_remaining = roster_df["Predicted Remaining"].sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total earned so far", f"{total_earned:.0f} pts")
        m2.metric("Total predicted remaining", f"{total_remaining:.0f} pts")
        m3.metric("Total projected season", f"{total_projected:.0f} pts",
                  delta=f"+{total_remaining:.0f} remaining")

        st.dataframe(roster_df, use_container_width=True, hide_index=True)

        # ── Swap suggestions ──────────────────────────────────────────────────
        st.subheader("Suggested Swaps")
        st.caption(
            "For each rostered player, the best available replacement at the same position "
            "ranked by projected season total."
        )

        rostered_ids    = set(st.session_state.roster)
        available_df    = all_players_df[~all_players_df["skaterFullName"].isin(rostered_ids)].copy()

        swap_rows = []
        for _, row in roster_df.iterrows():
            pos = row["Pos"]
            candidates = (
                available_df[available_df["positionCode"] == pos]
                .sort_values("predicted_season_total", ascending=False)
                .head(1)
            )
            if candidates.empty:
                continue
            best = candidates.iloc[0]
            gain = round(best["predicted_season_total"] - row["Projected Total"], 1)
            if gain > 0:
                swap_rows.append({
                    "Drop":             row["Player"],
                    "Pos":              pos,
                    "Drop earned":      row["Earned"],
                    "Drop projected":   row["Projected Total"],
                    "Pick up":          best["skaterFullName"],
                    "Pickup earned":    round(best["earned_so_far"], 1),
                    "Pickup projected": round(best["predicted_season_total"], 1),
                    "Projected gain":   gain,
                })

        if swap_rows:
            swap_df = (
                pd.DataFrame(swap_rows)
                .sort_values("Projected gain", ascending=False)
                .reset_index(drop=True)
            )
            # Highlight positive gains
            st.dataframe(
                swap_df.style.background_gradient(subset=["Projected gain"], cmap="RdYlGn"),
                use_container_width=True,
                hide_index=True,
            )
            # One-click swap
            st.caption("Click a row's player names above to identify the swap, then use the roster manager to apply it.")
        else:
            st.success("Your roster looks optimal — no beneficial swaps found at this time.")

# ── Cache controls (sidebar) ───────────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.subheader("🗄️ Cache")
if st.sidebar.button("Refresh current season"):
    CURRENT_CACHE.unlink(missing_ok=True)
    st.cache_data.clear()
    st.rerun()
if st.sidebar.button("Refresh all data + retrain"):
    TRAIN_CACHE.unlink(missing_ok=True)
    SCORE_CACHE.unlink(missing_ok=True)
    CURRENT_CACHE.unlink(missing_ok=True)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
