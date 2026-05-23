import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import json
import plotly.express as px
import numpy as np
import hashlib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


st.set_page_config(
    page_title="IPL Player Intelligence Dashboard",
    layout="wide"
)

st.title("IPL Player Intelligence Dashboard")
st.write("Real IPL ball-by-ball data from Cricsheet")


# -------------------------
# Load IPL Data
# -------------------------

@st.cache_data(show_spinner=False)
def load_ipl_data():
    url = "https://cricsheet.org/downloads/ipl_json.zip"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(response.content))

    matches_data = []
    deliveries_data = []

    for file_name in z.namelist():
        if not file_name.endswith(".json"):
            continue

        match_id = file_name.replace(".json", "")

        with z.open(file_name) as f:
            match = json.load(f)

        info = match.get("info", {})

        venue = info.get("venue")
        dates = info.get("dates", [])
        match_date = dates[0] if dates else None

        teams = info.get("teams", [])
        team1 = teams[0] if len(teams) > 0 else None
        team2 = teams[1] if len(teams) > 1 else None

        toss = info.get("toss", {})
        toss_winner = toss.get("winner")
        toss_decision = toss.get("decision")

        outcome = info.get("outcome", {})
        winner = outcome.get("winner")

        season = info.get("season")

        matches_data.append({
            "match_id": match_id,
            "date": match_date,
            "season": season,
            "venue": venue,
            "team1": team1,
            "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "winner": winner
        })

        innings_list = match.get("innings", [])

        for innings_no, innings in enumerate(innings_list, start=1):
            batting_team = innings.get("team")

            for over in innings.get("overs", []):
                over_no = over.get("over")

                for ball_no, delivery in enumerate(over.get("deliveries", []), start=1):
                    batter = delivery.get("batter")
                    bowler = delivery.get("bowler")
                    non_striker = delivery.get("non_striker")

                    runs = delivery.get("runs", {})
                    batter_runs = runs.get("batter", 0)
                    extras = runs.get("extras", 0)
                    total_runs = runs.get("total", 0)

                    wickets = delivery.get("wickets", [])
                    is_wicket = 1 if wickets else 0

                    dismissal_kind = None
                    player_out = None

                    if wickets:
                        dismissal_kind = wickets[0].get("kind")
                        player_out = wickets[0].get("player_out")

                    deliveries_data.append({
                        "match_id": match_id,
                        "innings": innings_no,
                        "batting_team": batting_team,
                        "over": over_no,
                        "ball": ball_no,
                        "batter": batter,
                        "non_striker": non_striker,
                        "bowler": bowler,
                        "batter_runs": batter_runs,
                        "extras": extras,
                        "total_runs": total_runs,
                        "is_wicket": is_wicket,
                        "dismissal_kind": dismissal_kind,
                        "player_out": player_out
                    })

    matches_df = pd.DataFrame(matches_data)
    deliveries_df = pd.DataFrame(deliveries_data)

    matches_df["date"] = pd.to_datetime(matches_df["date"], errors="coerce")
    matches_df["year"] = matches_df["date"].dt.year

    full_df = deliveries_df.merge(
        matches_df[
            [
                "match_id",
                "date",
                "year",
                "season",
                "venue",
                "team1",
                "team2",
                "toss_winner",
                "toss_decision",
                "winner"
            ]
        ],
        on="match_id",
        how="left"
    )

    return matches_df, deliveries_df, full_df


# -------------------------
# ML Helper Functions
# -------------------------

def get_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_selected_model_family():
    """Return the algorithm family selected in the Streamlit sidebar."""
    return st.session_state.get("active_model_family", "Random Forest")


def build_rf_regression_model(df, feature_cols, target_col):
    """
    Build a regression model using the selected algorithm family.
    The existing function name is retained to preserve all working page logic.
    """
    X = df[feature_cols]
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", get_onehot_encoder(), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    if get_selected_model_family() == "Linear / Logistic Regression":
        estimator = LinearRegression()
    else:
        estimator = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=12,
            min_samples_split=3
        )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    model.fit(X, y)
    return model


def build_rf_classification_model(df, feature_cols, target_col):
    """
    Build a classification model using the selected algorithm family.
    Random Forest is used when selected; Logistic Regression is used otherwise.
    """
    X = df[feature_cols]
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", get_onehot_encoder(), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    if get_selected_model_family() == "Linear / Logistic Regression":
        estimator = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=12,
            min_samples_split=3,
            class_weight="balanced"
        )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    model.fit(X, y)
    return model


# -------------------------
# Advanced Match Simulation Helper Functions
# -------------------------

SIMULATION_OUTCOMES = ["0", "1", "2", "3", "4", "6", "W"]


@st.cache_data(show_spinner=False)
def build_prediction_assets(all_deliveries, all_matches):
    """Prepare player ratings, team-player history, and delivery outcomes."""
    work = all_deliveries.copy()

    work["bowling_team"] = np.where(
        work["batting_team"] == work["team1"],
        work["team2"],
        work["team1"]
    )

    non_bowler_wickets = ["run out", "retired hurt", "obstructing the field"]
    work["bowler_wicket"] = np.where(
        (work["is_wicket"] == 1) &
        (~work["dismissal_kind"].isin(non_bowler_wickets)),
        1,
        0
    )

    work["phase"] = pd.cut(
        work["over"],
        bins=[-1, 5, 14, 100],
        labels=["Powerplay", "Middle", "Death"]
    ).astype(str)

    def make_outcome(row):
        if row["is_wicket"] == 1:
            return "W"
        runs = int(row["total_runs"])
        if runs >= 6:
            return "6"
        if runs == 4:
            return "4"
        if runs == 3:
            return "3"
        if runs == 2:
            return "2"
        if runs == 1:
            return "1"
        return "0"

    work["sim_outcome"] = work.apply(make_outcome, axis=1)

    batter_match = (
        work.groupby(["match_id", "batting_team", "batter"])
        .agg(
            runs=("batter_runs", "sum"),
            balls=("ball", "count")
        )
        .reset_index()
    )

    batter_rating = (
        batter_match.groupby("batter")
        .agg(
            batting_innings=("match_id", "count"),
            average_runs=("runs", "mean"),
            average_balls=("balls", "mean"),
            total_runs=("runs", "sum"),
            total_balls=("balls", "sum")
        )
        .reset_index()
        .rename(columns={"batter": "player"})
    )

    batter_rating["strike_rate"] = np.where(
        batter_rating["total_balls"] > 0,
        batter_rating["total_runs"] / batter_rating["total_balls"] * 100,
        0
    )

    batter_rating["batting_strength"] = (
        batter_rating["average_runs"] +
        batter_rating["strike_rate"] * 0.05
    )

    bowler_match = (
        work.groupby(["match_id", "bowling_team", "bowler"])
        .agg(
            wickets=("bowler_wicket", "sum"),
            runs_conceded=("total_runs", "sum"),
            balls=("ball", "count")
        )
        .reset_index()
    )

    bowler_rating = (
        bowler_match.groupby("bowler")
        .agg(
            bowling_innings=("match_id", "count"),
            average_wickets=("wickets", "mean"),
            total_wickets=("wickets", "sum"),
            total_runs_conceded=("runs_conceded", "sum"),
            total_balls_bowled=("balls", "sum")
        )
        .reset_index()
        .rename(columns={"bowler": "player"})
    )

    bowler_rating["economy"] = np.where(
        bowler_rating["total_balls_bowled"] > 0,
        bowler_rating["total_runs_conceded"] / bowler_rating["total_balls_bowled"] * 6,
        12
    )

    bowler_rating["bowling_strength"] = (
        bowler_rating["average_wickets"] * 15 +
        np.maximum(0, 10 - bowler_rating["economy"]) * 2
    )

    ratings = batter_rating.merge(
        bowler_rating[
            [
                "player",
                "bowling_innings",
                "average_wickets",
                "total_wickets",
                "economy",
                "bowling_strength"
            ]
        ],
        on="player",
        how="outer"
    ).fillna(0)

    batting_players = (
        work[["match_id", "batting_team", "batter"]]
        .rename(columns={"batting_team": "team", "batter": "player"})
    )

    bowling_players = (
        work[["match_id", "bowling_team", "bowler"]]
        .rename(columns={"bowling_team": "team", "bowler": "player"})
    )

    match_players = pd.concat(
        [batting_players, bowling_players],
        ignore_index=True
    ).dropna().drop_duplicates()

    team_players = (
        match_players.groupby(["team", "player"])
        .agg(matches_played=("match_id", "nunique"))
        .reset_index()
        .merge(
            ratings[
                ["player", "batting_strength", "bowling_strength", "total_runs", "total_wickets"]
            ],
            on="player",
            how="left"
        )
        .fillna(0)
        .sort_values(["team", "matches_played"], ascending=[True, False])
    )

    return work, ratings, match_players, team_players


def get_team_player_options(team, team_players):
    team_df = team_players[team_players["team"] == team].sort_values(
        ["matches_played", "batting_strength", "bowling_strength"],
        ascending=False
    )
    return team_df["player"].dropna().tolist()


def get_default_xi(team, team_players):
    return get_team_player_options(team, team_players)[:11]


def get_default_bowlers(team, selected_xi, team_players):
    team_df = team_players[
        (team_players["team"] == team) &
        (team_players["player"].isin(selected_xi))
    ].sort_values(
        ["bowling_strength", "total_wickets", "matches_played"],
        ascending=False
    )

    bowlers = team_df["player"].tolist()[:5]

    for player in selected_xi:
        if player not in bowlers and len(bowlers) < 5:
            bowlers.append(player)

    return bowlers


def xi_strength(selected_xi, ratings):
    selected = ratings[ratings["player"].isin(selected_xi)]

    return (
        float(selected["batting_strength"].sum()),
        float(selected["bowling_strength"].sum())
    )


@st.cache_data(show_spinner=False)
def build_advanced_match_training_data(all_deliveries, all_matches):
    work, ratings, match_players, _ = build_prediction_assets(all_deliveries, all_matches)

    player_ratings = ratings[["player", "batting_strength", "bowling_strength"]]
    match_strength = (
        match_players.merge(player_ratings, on="player", how="left")
        .fillna(0)
        .groupby(["match_id", "team"])
        .agg(
            batting_strength=("batting_strength", "sum"),
            bowling_strength=("bowling_strength", "sum")
        )
        .reset_index()
    )

    strength_lookup = {
        (row["match_id"], row["team"]): (
            row["batting_strength"], row["bowling_strength"]
        )
        for _, row in match_strength.iterrows()
    }

    rows = []
    valid_matches = all_matches.dropna(subset=["winner", "team1", "team2"]).copy()

    for _, match in valid_matches.iterrows():
        team1 = match["team1"]
        team2 = match["team2"]
        winner = match["winner"]

        if winner not in [team1, team2]:
            continue

        for team_a, team_b in [(team1, team2), (team2, team1)]:
            bat_a, bowl_a = strength_lookup.get((match["match_id"], team_a), (0, 0))
            bat_b, bowl_b = strength_lookup.get((match["match_id"], team_b), (0, 0))

            rows.append({
                "year": int(match["year"]) if pd.notna(match["year"]) else 0,
                "venue": match["venue"] if pd.notna(match["venue"]) else "Unknown",
                "team_a": team_a,
                "team_b": team_b,
                "toss_winner": match["toss_winner"] if pd.notna(match["toss_winner"]) else "Unknown",
                "toss_decision": match["toss_decision"] if pd.notna(match["toss_decision"]) else "Unknown",
                "batting_strength_a": bat_a,
                "bowling_strength_a": bowl_a,
                "batting_strength_b": bat_b,
                "bowling_strength_b": bowl_b,
                "team_a_win": 1 if winner == team_a else 0
            })

    return pd.DataFrame(rows), ratings


@st.cache_data(show_spinner=False)
def build_simulation_distributions(all_deliveries, all_matches):
    work, _, _, _ = build_prediction_assets(all_deliveries, all_matches)

    global_counts = (
        work["sim_outcome"]
        .value_counts()
        .reindex(SIMULATION_OUTCOMES, fill_value=0)
        .astype(float)
    )

    batter_counts = pd.crosstab(
        work["batter"],
        work["sim_outcome"]
    ).reindex(columns=SIMULATION_OUTCOMES, fill_value=0)

    bowler_counts = pd.crosstab(
        work["bowler"],
        work["sim_outcome"]
    ).reindex(columns=SIMULATION_OUTCOMES, fill_value=0)

    venue_phase_counts = pd.crosstab(
        [work["venue"].fillna("Unknown"), work["phase"]],
        work["sim_outcome"]
    ).reindex(columns=SIMULATION_OUTCOMES, fill_value=0)

    return global_counts, batter_counts, bowler_counts, venue_phase_counts


def get_ball_probabilities(batter, bowler, venue, phase, distributions):
    global_counts, batter_counts, bowler_counts, venue_phase_counts = distributions

    counts = global_counts.copy() + 2.0

    if batter in batter_counts.index:
        counts = counts + batter_counts.loc[batter] * 2.5

    if bowler in bowler_counts.index:
        counts = counts + bowler_counts.loc[bowler] * 2.5

    if (venue, phase) in venue_phase_counts.index:
        counts = counts + venue_phase_counts.loc[(venue, phase)] * 1.25

    probabilities = counts / counts.sum()
    return probabilities.values



# -------------------------
# Orange Cap / Purple Cap Prediction Helper Functions
# -------------------------

@st.cache_data(show_spinner=False)
def build_cap_season_tables(all_deliveries):
    """Create actual season totals used for Orange Cap and Purple Cap modelling."""
    work = all_deliveries.copy()
    work = work.dropna(subset=["year"])
    work["year"] = work["year"].astype(int)

    non_bowler_wickets = [
        "run out", "retired hurt", "retired out", "obstructing the field"
    ]
    work["bowler_wicket"] = np.where(
        (work["is_wicket"] == 1) &
        (~work["dismissal_kind"].isin(non_bowler_wickets)),
        1,
        0
    )

    orange = (
        work.groupby(["year", "batter"])
        .agg(
            runs=("batter_runs", "sum"),
            balls=("ball", "count"),
            matches=("match_id", "nunique")
        )
        .reset_index()
        .rename(columns={"batter": "player"})
    )
    orange["strike_rate"] = np.where(
        orange["balls"] > 0,
        orange["runs"] / orange["balls"] * 100,
        0
    )

    purple = (
        work.groupby(["year", "bowler"])
        .agg(
            wickets=("bowler_wicket", "sum"),
            balls=("ball", "count"),
            runs_conceded=("total_runs", "sum"),
            matches=("match_id", "nunique")
        )
        .reset_index()
        .rename(columns={"bowler": "player"})
    )
    purple["economy"] = np.where(
        purple["balls"] > 0,
        purple["runs_conceded"] / purple["balls"] * 6,
        0
    )

    return orange, purple


def make_cap_forecast_frames(season_df, value_col, rate_col, latest_year, target_year):
    """Create no-leakage training rows and next-season candidate features."""
    feature_cols = [
        "prediction_year",
        "seasons_played",
        "last_total",
        "last_matches",
        "last_rate",
        "career_avg_total",
        "best_total",
        "rolling_3_avg_total",
        "career_matches",
        "years_since_last"
    ]

    training_rows = []
    candidate_rows = []

    for player, group in season_df.groupby("player"):
        player_history = group.sort_values("year").reset_index(drop=True)

        for position in range(1, len(player_history)):
            previous = player_history.iloc[:position]
            current = player_history.iloc[position]
            last = previous.iloc[-1]
            totals = previous[value_col]

            row = {
                "player": player,
                "prediction_year": int(current["year"]),
                "seasons_played": len(previous),
                "last_total": float(last[value_col]),
                "last_matches": float(last["matches"]),
                "last_rate": float(last[rate_col]),
                "career_avg_total": float(totals.mean()),
                "best_total": float(totals.max()),
                "rolling_3_avg_total": float(totals.tail(3).mean()),
                "career_matches": float(previous["matches"].sum()),
                "years_since_last": int(current["year"] - last["year"]),
                "target": float(current[value_col])
            }
            training_rows.append(row)

        if int(player_history.iloc[-1]["year"]) == latest_year:
            last = player_history.iloc[-1]
            totals = player_history[value_col]
            candidate_rows.append({
                "player": player,
                "prediction_year": int(target_year),
                "seasons_played": len(player_history),
                "last_total": float(last[value_col]),
                "last_matches": float(last["matches"]),
                "last_rate": float(last[rate_col]),
                "career_avg_total": float(totals.mean()),
                "best_total": float(totals.max()),
                "rolling_3_avg_total": float(totals.tail(3).mean()),
                "career_matches": float(player_history["matches"].sum()),
                "years_since_last": int(target_year - latest_year)
            })

    return pd.DataFrame(training_rows), pd.DataFrame(candidate_rows), feature_cols


def predict_cap_table(season_df, value_col, rate_col, latest_year, target_year):
    training_df, candidates, feature_cols = make_cap_forecast_frames(
        season_df, value_col, rate_col, latest_year, target_year
    )

    if training_df.empty or candidates.empty:
        return pd.DataFrame(), training_df

    model = RandomForestRegressor(
        n_estimators=350,
        random_state=42,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2
    )
    model.fit(training_df[feature_cols], training_df["target"])
    candidates = candidates.copy()
    candidates[f"predicted_{value_col}"] = np.maximum(
        0,
        model.predict(candidates[feature_cols])
    )
    candidates[f"predicted_{value_col}"] = candidates[f"predicted_{value_col}"].round(0).astype(int)
    return candidates.sort_values(f"predicted_{value_col}", ascending=False), training_df


@st.cache_data(show_spinner=False)
def prepare_fixture_projection_data(all_deliveries):
    """Prepare team, opponent, venue and player performance data for future-fixture cap forecasts."""
    work = all_deliveries.copy()
    work = work.dropna(subset=["year"]).copy()
    work["year"] = work["year"].astype(int)
    work["bowling_team"] = np.where(
        work["batting_team"] == work["team1"], work["team2"], work["team1"]
    )
    non_bowler_wickets = [
        "run out", "retired hurt", "retired out", "obstructing the field"
    ]
    work["bowler_wicket"] = np.where(
        (work["is_wicket"] == 1) & (~work["dismissal_kind"].isin(non_bowler_wickets)),
        1,
        0
    )
    return work


def _safe_rate(numerator, denominator, fallback=0.0):
    return float(numerator / denominator) if denominator and denominator > 0 else float(fallback)


def _clipped_factor(specific_rate, overall_rate, low=0.80, high=1.20):
    if overall_rate <= 0 or specific_rate <= 0:
        return 1.0
    return float(np.clip(specific_rate / overall_rate, low, high))


def project_caps_from_future_fixtures(all_deliveries, recorded_year, fixtures, squads, impacts):
    """
    Correct season-in-progress cap forecast.

    The recorded Orange/Purple standings for the chosen season are the fixed baseline.
    Only selected players in selected remaining fixtures receive additional expected
    runs/wickets. Players without a future selected fixture remain in the final table
    with their already-recorded totals unchanged.
    """
    work = prepare_fixture_projection_data(all_deliveries)
    recorded_year = int(recorded_year)
    current = work[work["year"] == recorded_year].copy()
    history = work[work["year"] <= recorded_year].copy()
    recent = history[history["year"] >= recorded_year - 2].copy()
    if recent.empty:
        recent = history.copy()

    # -------- Fixed baseline: every player already present in actual season standings --------
    orange_base = (
        current.groupby("batter")
        .agg(
            Recorded_Runs=("batter_runs", "sum"),
            Recorded_Matches=("match_id", "nunique")
        )
        .reset_index()
        .rename(columns={"batter": "Player"})
    )
    purple_base = (
        current.groupby("bowler")
        .agg(
            Recorded_Wickets=("bowler_wicket", "sum"),
            Recorded_Matches=("match_id", "nunique")
        )
        .reset_index()
        .rename(columns={"bowler": "Player"})
    )

    current_batter_team = (
        current.groupby(["batter", "batting_team"])["match_id"]
        .nunique().reset_index(name="appearances")
        .sort_values(["batter", "appearances"], ascending=[True, False])
        .drop_duplicates("batter")
        .set_index("batter")["batting_team"].to_dict()
    )
    current_bowler_team = (
        current.groupby(["bowler", "bowling_team"])["match_id"]
        .nunique().reset_index(name="appearances")
        .sort_values(["bowler", "appearances"], ascending=[True, False])
        .drop_duplicates("bowler")
        .set_index("bowler")["bowling_team"].to_dict()
    )

    orange = {
        row["Player"]: {
            "Player": row["Player"],
            "Team": current_batter_team.get(row["Player"], "Not selected for remaining fixture"),
            "Recorded Runs": int(row["Recorded_Runs"]),
            "Recorded Matches": int(row["Recorded_Matches"]),
            "Remaining Matches": 0,
            "Projected Additional Runs": 0.0,
            "Impact Option": "No",
            "Projection Basis": "Actual standings only"
        }
        for _, row in orange_base.iterrows()
    }
    purple = {
        row["Player"]: {
            "Player": row["Player"],
            "Team": current_bowler_team.get(row["Player"], "Not selected for remaining fixture"),
            "Recorded Wickets": int(row["Recorded_Wickets"]),
            "Recorded Matches": int(row["Recorded_Matches"]),
            "Remaining Matches": 0,
            "Projected Additional Wickets": 0.0,
            "Impact Option": "No",
            "Projection Basis": "Actual standings only"
        }
        for _, row in purple_base.iterrows()
    }

    # Helper: form must use this season first; older history is only a fallback for a selected player.
    def expected_for_fixture(player, team, opponent, venue):
        season_bat = current[(current["batter"] == player) & (current["batting_team"] == team)]
        season_bowl = current[(current["bowler"] == player) & (current["bowling_team"] == team)]
        recent_bat = recent[(recent["batter"] == player) & (recent["batting_team"] == team)]
        recent_bowl = recent[(recent["bowler"] == player) & (recent["bowling_team"] == team)]
        if recent_bat.empty:
            recent_bat = recent[recent["batter"] == player]
        if recent_bowl.empty:
            recent_bowl = recent[recent["bowler"] == player]

        bat_source = season_bat if season_bat["match_id"].nunique() > 0 else recent_bat
        bowl_source = season_bowl if season_bowl["match_id"].nunique() > 0 else recent_bowl
        basis = "Current season form" if season_bat["match_id"].nunique() > 0 or season_bowl["match_id"].nunique() > 0 else "Recent-history fallback"

        bat_matches = bat_source["match_id"].nunique()
        bowl_matches = bowl_source["match_id"].nunique()
        base_runs = _safe_rate(bat_source["batter_runs"].sum(), bat_matches, 0.0)
        base_wickets = _safe_rate(bowl_source["bowler_wicket"].sum(), bowl_matches, 0.0)

        all_player_bat = history[history["batter"] == player]
        all_player_bowl = history[history["bowler"] == player]
        venue_bat = all_player_bat[all_player_bat["venue"] == venue]
        opp_bat = all_player_bat[all_player_bat["bowling_team"] == opponent]
        venue_bowl = all_player_bowl[all_player_bowl["venue"] == venue]
        opp_bowl = all_player_bowl[all_player_bowl["batting_team"] == opponent]

        venue_run_rate = _safe_rate(venue_bat["batter_runs"].sum(), venue_bat["match_id"].nunique(), base_runs)
        opp_run_rate = _safe_rate(opp_bat["batter_runs"].sum(), opp_bat["match_id"].nunique(), base_runs)
        venue_wicket_rate = _safe_rate(venue_bowl["bowler_wicket"].sum(), venue_bowl["match_id"].nunique(), base_wickets)
        opp_wicket_rate = _safe_rate(opp_bowl["bowler_wicket"].sum(), opp_bowl["match_id"].nunique(), base_wickets)

        # Only small fixture adjustments: prevents inflated/double-looking totals.
        run_factor = _clipped_factor(venue_run_rate, base_runs, 0.95, 1.05) * _clipped_factor(opp_run_rate, base_runs, 0.95, 1.05)
        wicket_factor = _clipped_factor(venue_wicket_rate, base_wickets, 0.95, 1.05) * _clipped_factor(opp_wicket_rate, base_wickets, 0.95, 1.05)

        expected_runs = min(max(base_runs * run_factor, 0.0), 90.0)
        expected_wickets = min(max(base_wickets * wicket_factor, 0.0), 4.0)
        return expected_runs, expected_wickets, basis

    # -------- Add only future-match contributions for chosen squads --------
    for _, fixture in fixtures.iterrows():
        team_1, team_2, venue = fixture["Team 1"], fixture["Team 2"], fixture["Venue"]
        for team, opponent in [(team_1, team_2), (team_2, team_1)]:
            squad = squads.get(team, [])
            impact_player = impacts.get(team)
            for player in squad:
                if player not in orange:
                    orange[player] = {
                        "Player": player, "Team": team, "Recorded Runs": 0,
                        "Recorded Matches": 0, "Remaining Matches": 0,
                        "Projected Additional Runs": 0.0, "Impact Option": "No",
                        "Projection Basis": "No current-season runs recorded"
                    }
                if player not in purple:
                    purple[player] = {
                        "Player": player, "Team": team, "Recorded Wickets": 0,
                        "Recorded Matches": 0, "Remaining Matches": 0,
                        "Projected Additional Wickets": 0.0, "Impact Option": "No",
                        "Projection Basis": "No current-season wickets recorded"
                    }

                exp_runs, exp_wickets, basis = expected_for_fixture(player, team, opponent, venue)
                participation = 0.60 if player == impact_player else 1.0
                exp_runs *= participation
                exp_wickets *= participation

                orange[player]["Team"] = team
                orange[player]["Remaining Matches"] += 1
                orange[player]["Projected Additional Runs"] += exp_runs
                orange[player]["Impact Option"] = "Yes" if player == impact_player else orange[player]["Impact Option"]
                orange[player]["Projection Basis"] = basis

                purple[player]["Team"] = team
                purple[player]["Remaining Matches"] += 1
                purple[player]["Projected Additional Wickets"] += exp_wickets
                purple[player]["Impact Option"] = "Yes" if player == impact_player else purple[player]["Impact Option"]
                purple[player]["Projection Basis"] = basis

    orange_projection = pd.DataFrame(list(orange.values()))
    purple_projection = pd.DataFrame(list(purple.values()))
    if orange_projection.empty or purple_projection.empty:
        return orange_projection, purple_projection

    orange_projection["Projected Additional Runs"] = orange_projection["Projected Additional Runs"].round(1)
    orange_projection["Projected Final Runs"] = (
        orange_projection["Recorded Runs"] + orange_projection["Projected Additional Runs"]
    ).round(0).astype(int)
    orange_projection["Recorded Rank"] = orange_projection["Recorded Runs"].rank(method="min", ascending=False).astype(int)
    orange_projection = orange_projection.sort_values(
        ["Projected Final Runs", "Recorded Runs"], ascending=False
    ).reset_index(drop=True)
    orange_projection.insert(0, "Projected Rank", range(1, len(orange_projection) + 1))

    purple_projection["Projected Additional Wickets"] = purple_projection["Projected Additional Wickets"].round(2)
    purple_projection["Projected Final Wickets"] = (
        purple_projection["Recorded Wickets"] + purple_projection["Projected Additional Wickets"]
    ).round(0).astype(int)
    purple_projection["Recorded Rank"] = purple_projection["Recorded Wickets"].rank(method="min", ascending=False).astype(int)
    purple_projection = purple_projection.sort_values(
        ["Projected Final Wickets", "Recorded Wickets"], ascending=False
    ).reset_index(drop=True)
    purple_projection.insert(0, "Projected Rank", range(1, len(purple_projection) + 1))
    return orange_projection, purple_projection


def apply_impact_replacement(starting_xi, impact_player, replaced_player):
    """Return the active simulated XI after a user-selected Impact Player substitution."""
    active_xi = list(starting_xi)
    if impact_player and impact_player != "No Impact Player" and replaced_player in active_xi:
        active_xi[active_xi.index(replaced_player)] = impact_player
    return active_xi

def overs_text(balls):
    return f"{int(balls // 6)}.{int(balls % 6)}"


def choose_bowler(bowlers, bowling_stats, previous_bowler):
    eligible = [
        bowler for bowler in bowlers
        if bowling_stats[bowler]["balls"] < 24 and bowler != previous_bowler
    ]

    if not eligible:
        eligible = [
            bowler for bowler in bowlers
            if bowling_stats[bowler]["balls"] < 24
        ]

    if not eligible:
        return None

    return sorted(
        eligible,
        key=lambda name: (bowling_stats[name]["balls"], bowlers.index(name))
    )[0]


def simulate_innings(
    batting_team,
    bowling_team,
    batting_xi,
    bowling_xi,
    bowlers,
    venue,
    innings_number,
    distributions,
    rng,
    target=None
):
    batting_stats = {
        batter: {
            "Batter": batter,
            "Runs": 0,
            "Balls": 0,
            "Fours": 0,
            "Sixes": 0,
            "Dismissal": "Did not bat"
        }
        for batter in batting_xi
    }

    bowling_stats = {
        bowler: {
            "Bowler": bowler,
            "balls": 0,
            "Runs": 0,
            "Wickets": 0
        }
        for bowler in bowlers
    }

    score = 0
    wickets = 0
    legal_balls = 0
    next_batter_index = 2
    striker = batting_xi[0]
    non_striker = batting_xi[1]
    batting_stats[striker]["Dismissal"] = "Not Out"
    batting_stats[non_striker]["Dismissal"] = "Not Out"

    ball_log = []
    fall_of_wickets = []
    previous_bowler = None

    for over_number in range(20):
        if wickets >= 10 or (target is not None and score >= target):
            break

        bowler = choose_bowler(bowlers, bowling_stats, previous_bowler)
        if bowler is None:
            break

        previous_bowler = bowler

        for ball_in_over in range(1, 7):
            if wickets >= 10 or (target is not None and score >= target):
                break

            phase = "Powerplay" if over_number <= 5 else "Middle" if over_number <= 14 else "Death"
            probabilities = get_ball_probabilities(
                striker,
                bowler,
                venue,
                phase,
                distributions
            )

            outcome = rng.choice(SIMULATION_OUTCOMES, p=probabilities)
            legal_balls += 1
            batting_stats[striker]["Balls"] += 1
            bowling_stats[bowler]["balls"] += 1

            if outcome == "W":
                wickets += 1
                bowling_stats[bowler]["Wickets"] += 1
                batting_stats[striker]["Dismissal"] = f"b {bowler}"
                fall_of_wickets.append(
                    f"{score}/{wickets} ({striker}, {overs_text(legal_balls)})"
                )
                description = f"WICKET - {striker} b {bowler}"

                if wickets < 10 and next_batter_index < len(batting_xi):
                    striker = batting_xi[next_batter_index]
                    batting_stats[striker]["Dismissal"] = "Not Out"
                    next_batter_index += 1

            else:
                runs = int(outcome)
                score += runs
                batting_stats[striker]["Runs"] += runs
                bowling_stats[bowler]["Runs"] += runs

                if runs == 4:
                    batting_stats[striker]["Fours"] += 1
                elif runs == 6:
                    batting_stats[striker]["Sixes"] += 1

                description = f"{striker} scored {runs} run{'s' if runs != 1 else ''}"

                if runs % 2 == 1:
                    striker, non_striker = non_striker, striker

            ball_log.append({
                "Innings": innings_number,
                "Over": f"{over_number + 1}.{ball_in_over}",
                "Batting Team": batting_team,
                "Bowler": bowler,
                "Event": outcome,
                "Score": score,
                "Wickets": wickets,
                "Description": description
            })

        if wickets < 10 and (target is None or score < target):
            striker, non_striker = non_striker, striker

    batting_df = pd.DataFrame(batting_stats.values())
    batted_mask = batting_df["Dismissal"] != "Did not bat"
    batting_df.loc[batted_mask, "Strike Rate"] = (
        batting_df.loc[batted_mask, "Runs"] /
        batting_df.loc[batted_mask, "Balls"].replace(0, np.nan) * 100
    ).fillna(0).round(2)
    batting_df.loc[~batted_mask, "Strike Rate"] = 0

    bowling_df = pd.DataFrame(bowling_stats.values())
    bowling_df["Overs"] = bowling_df["balls"].apply(overs_text)
    bowling_df["Economy"] = np.where(
        bowling_df["balls"] > 0,
        bowling_df["Runs"] / bowling_df["balls"] * 6,
        0
    ).round(2)
    bowling_df = bowling_df[
        ["Bowler", "Overs", "Runs", "Wickets", "Economy"]
    ].sort_values(["Wickets", "Runs"], ascending=[False, True])

    return {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "score": int(score),
        "wickets": int(wickets),
        "balls": int(legal_balls),
        "overs": overs_text(legal_balls),
        "batting": batting_df,
        "bowling": bowling_df,
        "ball_log": pd.DataFrame(ball_log),
        "fall_of_wickets": fall_of_wickets
    }


def match_setup_inputs(key_prefix, all_matches, team_players):
    teams = sorted(
        pd.concat([all_matches["team1"], all_matches["team2"]])
        .dropna()
        .unique()
    )
    venues = sorted(all_matches["venue"].dropna().unique())

    col1, col2 = st.columns(2)

    with col1:
        team1 = st.selectbox("Select Team 1", teams, key=f"{key_prefix}_team1")
        team2_options = [team for team in teams if team != team1]
        team2 = st.selectbox("Select Team 2", team2_options, key=f"{key_prefix}_team2")
        venue = st.selectbox("Select Venue", venues, key=f"{key_prefix}_venue")

    with col2:
        toss_winner = st.selectbox(
            "Select Toss Winner",
            [team1, team2],
            key=f"{key_prefix}_toss_winner"
        )
        toss_decision = st.selectbox(
            "Select Toss Decision",
            ["bat", "field"],
            key=f"{key_prefix}_toss_decision"
        )
        prediction_year = st.selectbox(
            "Prediction Year",
            sorted(all_matches["year"].dropna().astype(int).unique(), reverse=True),
            key=f"{key_prefix}_year"
        )

    team1_players = get_team_player_options(team1, team_players)
    team2_players = get_team_player_options(team2, team_players)

    st.subheader("Select Playing XI in Batting Order")
    st.caption("Keep the players in the desired batting-order sequence. Select exactly 11 players for each team.")

    xi_col1, xi_col2 = st.columns(2)
    with xi_col1:
        xi1 = st.multiselect(
            f"{team1} Playing XI",
            team1_players,
            default=get_default_xi(team1, team_players),
            key=f"{key_prefix}_xi1"
        )
    with xi_col2:
        xi2 = st.multiselect(
            f"{team2} Playing XI",
            team2_players,
            default=get_default_xi(team2, team_players),
            key=f"{key_prefix}_xi2"
        )

    return team1, team2, venue, toss_winner, toss_decision, prediction_year, xi1, xi2


with st.spinner("Loading latest real IPL data from Cricsheet..."):
    matches_df, deliveries_df, full_df = load_ipl_data()

st.success("IPL data loaded successfully")


# -------------------------
# Global Year Filters
# -------------------------

st.sidebar.header("Global Filters")

available_years = sorted(full_df["year"].dropna().astype(int).unique())

year_filter_type = st.sidebar.radio(
    "Choose Year Filter Type",
    ["Single Year", "Year Range"]
)

if year_filter_type == "Single Year":
    default_year = 2022 if 2022 in available_years else available_years[0]

    selected_year = st.sidebar.selectbox(
        "Select IPL Year",
        available_years,
        index=available_years.index(default_year)
    )

    filtered_df = full_df[full_df["year"] == selected_year]
    filtered_matches_df = matches_df[matches_df["year"] == selected_year]

    st.sidebar.success(f"Showing IPL data for {selected_year}")

else:
    start_year = 2022 if 2022 in available_years else min(available_years)
    end_year = max(available_years)

    selected_year_range = st.sidebar.slider(
        "Select IPL Year Range",
        min_value=min(available_years),
        max_value=max(available_years),
        value=(start_year, end_year)
    )

    filtered_df = full_df[
        (full_df["year"] >= selected_year_range[0]) &
        (full_df["year"] <= selected_year_range[1])
    ]

    filtered_matches_df = matches_df[
        (matches_df["year"] >= selected_year_range[0]) &
        (matches_df["year"] <= selected_year_range[1])
    ]

    st.sidebar.success(
        f"Showing IPL data from {selected_year_range[0]} to {selected_year_range[1]}"
    )


# -------------------------
# Navigation
# -------------------------

page = st.sidebar.radio(
    "Choose Analysis",
    [
        "Dataset Overview",
        "Batter Analysis",
        "Bowler Analysis",
        "Batter vs Bowler",
        "Venue Analysis",
        "Match History",

        "ML - Batter vs Bowler Prediction",
        "ML - Match Winner Prediction",
        "ML - Bowler Runs Prediction",
        "ML - Batter Runs Prediction",
        "ML - Advanced Team Winner Prediction",
        "ML - Orange & Purple Cap Prediction",
        "ML - Full Match Scorecard Simulation",

        "Download Filtered Data"
    ]
)

model_selection_pages = {
    "ML - Batter vs Bowler Prediction",
    "ML - Match Winner Prediction",
    "ML - Bowler Runs Prediction",
    "ML - Batter Runs Prediction",
    "ML - Advanced Team Winner Prediction"
}

if page in model_selection_pages:
    st.sidebar.divider()
    selected_model_family = st.sidebar.selectbox(
        "Choose Prediction Algorithm",
        ["Random Forest", "Linear / Logistic Regression"],
        key="algorithm_family_selector",
        help=(
            "Numerical predictions use RandomForestRegressor or LinearRegression. "
            "Winner/dismissal predictions use RandomForestClassifier or LogisticRegression."
        )
    )
    st.session_state["active_model_family"] = selected_model_family
    if selected_model_family == "Random Forest":
        st.sidebar.caption("Regression: RandomForestRegressor | Classification: RandomForestClassifier")
    else:
        st.sidebar.caption("Regression: LinearRegression | Classification: LogisticRegression")
else:
    st.session_state["active_model_family"] = "Random Forest"


# -------------------------
# Dataset Overview
# -------------------------

if page == "Dataset Overview":
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Filtered Matches", filtered_matches_df["match_id"].nunique())
    col2.metric("Filtered Deliveries", len(filtered_df))

    total_players = pd.concat([
        filtered_df["batter"],
        filtered_df["bowler"]
    ]).dropna().nunique()

    col3.metric("Players", total_players)
    col4.metric("Latest Year in Data", int(filtered_df["year"].max()))

    st.subheader("Year-wise Matches")

    year_summary = (
        filtered_matches_df.groupby("year")
        .agg(matches=("match_id", "nunique"))
        .reset_index()
    )

    fig = px.bar(
        year_summary,
        x="year",
        y="matches",
        title="IPL Matches Available by Year",
        text="matches"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matches Data")
    st.dataframe(filtered_matches_df, use_container_width=True)

    st.subheader("Ball-by-Ball Deliveries Data")

    row_limit = st.number_input(
        "Number of rows to display",
        min_value=100,
        max_value=max(100, len(filtered_df)),
        value=min(1000, len(filtered_df)),
        step=100
    )

    st.dataframe(filtered_df.head(row_limit), use_container_width=True)


# -------------------------
# Batter Analysis
# -------------------------

elif page == "Batter Analysis":
    st.header("Batter Analysis")

    batters = sorted(filtered_df["batter"].dropna().unique())

    selected_batter = st.selectbox("Select Batter", batters)

    batter_df = filtered_df[filtered_df["batter"] == selected_batter]

    balls = len(batter_df)
    runs = batter_df["batter_runs"].sum()
    fours = (batter_df["batter_runs"] == 4).sum()
    sixes = (batter_df["batter_runs"] == 6).sum()
    outs = filtered_df[filtered_df["player_out"] == selected_batter].shape[0]

    strike_rate = round((runs / balls) * 100, 2) if balls > 0 else 0
    average = round(runs / outs, 2) if outs > 0 else runs

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    col1.metric("Runs", runs)
    col2.metric("Balls", balls)
    col3.metric("Fours", fours)
    col4.metric("Sixes", sixes)
    col5.metric("Outs", outs)
    col6.metric("Strike Rate", strike_rate)
    col7.metric("Average", average)

    st.subheader("Year-wise Batter Performance")

    batter_year = (
        batter_df.groupby("year")
        .agg(
            runs=("batter_runs", "sum"),
            balls=("batter", "count"),
            fours=("batter_runs", lambda x: (x == 4).sum()),
            sixes=("batter_runs", lambda x: (x == 6).sum())
        )
        .reset_index()
    )

    batter_year["strike_rate"] = round(
        batter_year["runs"] / batter_year["balls"] * 100,
        2
    )

    fig = px.line(
        batter_year,
        x="year",
        y="runs",
        markers=True,
        title=f"{selected_batter} Runs by Year"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Batter vs Bowlers")

    batter_vs_bowler = (
        batter_df.groupby("bowler")
        .agg(
            balls=("bowler", "count"),
            runs=("batter_runs", "sum"),
            fours=("batter_runs", lambda x: (x == 4).sum()),
            sixes=("batter_runs", lambda x: (x == 6).sum())
        )
        .reset_index()
    )

    wickets_against = (
        filtered_df[filtered_df["player_out"] == selected_batter]
        .groupby("bowler")
        .size()
        .reset_index(name="times_out")
    )

    batter_vs_bowler = batter_vs_bowler.merge(
        wickets_against,
        on="bowler",
        how="left"
    )

    batter_vs_bowler["times_out"] = batter_vs_bowler["times_out"].fillna(0).astype(int)

    batter_vs_bowler["strike_rate"] = round(
        batter_vs_bowler["runs"] / batter_vs_bowler["balls"] * 100,
        2
    )

    batter_vs_bowler = batter_vs_bowler.sort_values(
        by=["runs"],
        ascending=False
    )

    top_bowlers = batter_vs_bowler.head(15)

    fig = px.bar(
        top_bowlers,
        x="bowler",
        y="runs",
        title=f"Top Bowlers Faced by {selected_batter}",
        text="runs"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(batter_vs_bowler, use_container_width=True)

    st.subheader("Most Dismissed By")

    most_dismissed_by = (
        filtered_df[filtered_df["player_out"] == selected_batter]
        .groupby("bowler")
        .size()
        .reset_index(name="times_out")
        .sort_values("times_out", ascending=False)
    )

    st.dataframe(most_dismissed_by, use_container_width=True)


# -------------------------
# Bowler Analysis
# -------------------------

elif page == "Bowler Analysis":
    st.header("Bowler Analysis")

    bowlers = sorted(filtered_df["bowler"].dropna().unique())

    selected_bowler = st.selectbox("Select Bowler", bowlers)

    bowler_df = filtered_df[filtered_df["bowler"] == selected_bowler]

    balls = len(bowler_df)
    overs = round(balls / 6, 2)
    runs_conceded = bowler_df["total_runs"].sum()
    wickets = bowler_df["is_wicket"].sum()
    fours = (bowler_df["batter_runs"] == 4).sum()
    sixes = (bowler_df["batter_runs"] == 6).sum()

    economy = round((runs_conceded / balls) * 6, 2) if balls > 0 else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Balls Bowled", balls)
    col2.metric("Overs", overs)
    col3.metric("Runs Conceded", runs_conceded)
    col4.metric("Wickets", wickets)
    col5.metric("Economy", economy)
    col6.metric("Sixes Conceded", sixes)

    st.subheader("Year-wise Bowler Wickets")

    bowler_year = (
        bowler_df.groupby("year")
        .agg(
            wickets=("is_wicket", "sum"),
            runs_conceded=("total_runs", "sum"),
            balls=("bowler", "count")
        )
        .reset_index()
    )

    bowler_year["economy"] = round(
        bowler_year["runs_conceded"] / bowler_year["balls"] * 6,
        2
    )

    fig = px.line(
        bowler_year,
        x="year",
        y="wickets",
        markers=True,
        title=f"{selected_bowler} Wickets by Year"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Bowler vs Batters")

    bowler_vs_batter = (
        bowler_df.groupby("batter")
        .agg(
            balls=("batter", "count"),
            runs=("batter_runs", "sum"),
            fours=("batter_runs", lambda x: (x == 4).sum()),
            sixes=("batter_runs", lambda x: (x == 6).sum()),
            wickets=("is_wicket", "sum")
        )
        .reset_index()
    )

    bowler_vs_batter["strike_rate_against_bowler"] = round(
        bowler_vs_batter["runs"] / bowler_vs_batter["balls"] * 100,
        2
    )

    bowler_vs_batter = bowler_vs_batter.sort_values(
        by=["wickets", "runs"],
        ascending=[False, False]
    )

    top_batters = bowler_vs_batter.head(15)

    fig = px.bar(
        top_batters,
        x="batter",
        y="wickets",
        title=f"Top Wickets by {selected_bowler}",
        text="wickets"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(bowler_vs_batter, use_container_width=True)


# -------------------------
# Batter vs Bowler
# -------------------------

elif page == "Batter vs Bowler":
    st.header("Batter vs Bowler Matchup")

    batters = sorted(filtered_df["batter"].dropna().unique())
    bowlers = sorted(filtered_df["bowler"].dropna().unique())

    col1, col2 = st.columns(2)

    with col1:
        selected_batter = st.selectbox("Select Batter", batters)

    with col2:
        selected_bowler = st.selectbox("Select Bowler", bowlers)

    matchup_df = filtered_df[
        (filtered_df["batter"] == selected_batter) &
        (filtered_df["bowler"] == selected_bowler)
    ]

    balls = len(matchup_df)
    runs = matchup_df["batter_runs"].sum()
    fours = (matchup_df["batter_runs"] == 4).sum()
    sixes = (matchup_df["batter_runs"] == 6).sum()

    outs = filtered_df[
        (filtered_df["player_out"] == selected_batter) &
        (filtered_df["bowler"] == selected_bowler)
    ].shape[0]

    strike_rate = round((runs / balls) * 100, 2) if balls > 0 else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Balls", balls)
    col2.metric("Runs", runs)
    col3.metric("Fours", fours)
    col4.metric("Sixes", sixes)
    col5.metric("Times Out", outs)
    col6.metric("Strike Rate", strike_rate)

    if balls == 0:
        st.warning("No direct matchup data available for this batter and bowler in the selected year filter.")

    else:
        st.subheader("Graphical Matchup Summary")

        col1, col2 = st.columns(2)

        with col1:
            over_summary = (
                matchup_df.groupby("over")
                .agg(
                    balls=("ball", "count"),
                    runs=("batter_runs", "sum")
                )
                .reset_index()
            )

            fig = px.bar(
                over_summary,
                x="over",
                y="runs",
                title=f"Runs by Over: {selected_batter} vs {selected_bowler}",
                text="runs"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            run_type_summary = (
                matchup_df["batter_runs"]
                .value_counts()
                .reset_index()
            )

            run_type_summary.columns = ["runs_per_ball", "balls"]

            fig = px.bar(
                run_type_summary,
                x="runs_per_ball",
                y="balls",
                title="Ball Result Distribution",
                text="balls"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Year-wise Matchup")

        year_matchup = (
            matchup_df.groupby("year")
            .agg(
                balls=("ball", "count"),
                runs=("batter_runs", "sum"),
                fours=("batter_runs", lambda x: (x == 4).sum()),
                sixes=("batter_runs", lambda x: (x == 6).sum())
            )
            .reset_index()
        )

        year_matchup["strike_rate"] = round(
            year_matchup["runs"] / year_matchup["balls"] * 100,
            2
        )

        fig = px.line(
            year_matchup,
            x="year",
            y="strike_rate",
            markers=True,
            title=f"Strike Rate Trend: {selected_batter} vs {selected_bowler}"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Dismissal Details")

        dismissal_df = filtered_df[
            (filtered_df["player_out"] == selected_batter) &
            (filtered_df["bowler"] == selected_bowler)
        ][
            [
                "date",
                "year",
                "venue",
                "batting_team",
                "bowler",
                "dismissal_kind",
                "player_out"
            ]
        ]

        if dismissal_df.empty:
            st.info("This bowler has not dismissed this batter in the selected year filter.")
        else:
            st.dataframe(dismissal_df, use_container_width=True)

        st.subheader("Ball-by-Ball Matchup Data")

        st.dataframe(
            matchup_df[
                [
                    "date",
                    "year",
                    "venue",
                    "innings",
                    "batting_team",
                    "over",
                    "ball",
                    "batter",
                    "bowler",
                    "batter_runs",
                    "extras",
                    "total_runs",
                    "is_wicket",
                    "dismissal_kind",
                    "player_out"
                ]
            ],
            use_container_width=True
        )


# -------------------------
# Venue Analysis
# -------------------------

elif page == "Venue Analysis":
    st.header("Venue Analysis")

    venues = sorted(filtered_matches_df["venue"].dropna().unique())

    selected_venue = st.selectbox("Select Venue", venues)

    venue_matches = filtered_matches_df[filtered_matches_df["venue"] == selected_venue]
    venue_balls = filtered_df[filtered_df["venue"] == selected_venue]

    col1, col2, col3 = st.columns(3)

    col1.metric("Matches", venue_matches["match_id"].nunique())
    col2.metric("Total Runs", venue_balls["total_runs"].sum())
    col3.metric("Total Wickets", venue_balls["is_wicket"].sum())

    st.subheader("Team Wins at This Venue")

    venue_wins = (
        venue_matches.groupby("winner")
        .size()
        .reset_index(name="wins")
        .sort_values("wins", ascending=False)
    )

    fig = px.bar(
        venue_wins,
        x="winner",
        y="wins",
        title=f"Team Wins at {selected_venue}",
        text="wins"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(venue_wins, use_container_width=True)

    st.subheader("Batter Performance at This Venue")

    venue_batter_stats = (
        venue_balls.groupby("batter")
        .agg(
            balls=("batter", "count"),
            runs=("batter_runs", "sum"),
            fours=("batter_runs", lambda x: (x == 4).sum()),
            sixes=("batter_runs", lambda x: (x == 6).sum())
        )
        .reset_index()
    )

    venue_batter_stats["strike_rate"] = round(
        venue_batter_stats["runs"] / venue_batter_stats["balls"] * 100,
        2
    )

    venue_batter_stats = venue_batter_stats.sort_values(
        "runs",
        ascending=False
    )

    fig = px.bar(
        venue_batter_stats.head(20),
        x="batter",
        y="runs",
        title=f"Top Batters at {selected_venue}",
        text="runs"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(venue_batter_stats, use_container_width=True)


# -------------------------
# Match History
# -------------------------

elif page == "Match History":
    st.header("Match History")

    teams = sorted(
        pd.concat([filtered_matches_df["team1"], filtered_matches_df["team2"]])
        .dropna()
        .unique()
    )

    selected_team = st.selectbox("Select Team", teams)

    team_matches = filtered_matches_df[
        (filtered_matches_df["team1"] == selected_team) |
        (filtered_matches_df["team2"] == selected_team)
    ]

    col1, col2 = st.columns(2)

    col1.metric("Matches Played", len(team_matches))
    col2.metric("Matches Won", (team_matches["winner"] == selected_team).sum())

    st.subheader("Team Match History")

    st.dataframe(
        team_matches[
            [
                "date",
                "year",
                "venue",
                "team1",
                "team2",
                "toss_winner",
                "toss_decision",
                "winner"
            ]
        ].sort_values("date", ascending=False),
        use_container_width=True
    )


# -------------------------
# ML - Batter vs Bowler Prediction
# -------------------------

elif page == "ML - Batter vs Bowler Prediction":
    st.header("ML - Batter vs Bowler Prediction")
    st.write(f"Selected model family: **{selected_model_family}**. Numerical outputs use the corresponding regression model; dismissal uses the corresponding classifier.")

    ml_df = filtered_df.copy()

    if ml_df.empty:
        st.warning("No data available for selected year filter.")
    else:
        matchup_base = (
            ml_df.groupby(
                [
                    "match_id",
                    "year",
                    "venue",
                    "batting_team",
                    "team1",
                    "team2",
                    "batter",
                    "bowler"
                ]
            )
            .agg(
                balls_faced=("ball", "count"),
                runs_scored=("batter_runs", "sum"),
                fours=("batter_runs", lambda x: (x == 4).sum()),
                sixes=("batter_runs", lambda x: (x == 6).sum())
            )
            .reset_index()
        )

        dismissal_base = (
            ml_df[ml_df["player_out"].notna()]
            .groupby(["match_id", "batter", "bowler"])
            .agg(dismissed=("player_out", "count"))
            .reset_index()
        )

        matchup_base = matchup_base.merge(
            dismissal_base,
            on=["match_id", "batter", "bowler"],
            how="left"
        )

        matchup_base["dismissed"] = matchup_base["dismissed"].fillna(0)
        matchup_base["dismissed"] = matchup_base["dismissed"].apply(lambda x: 1 if x > 0 else 0)

        for col in ["venue", "batting_team", "team1", "team2", "batter", "bowler"]:
            matchup_base[col] = matchup_base[col].fillna("Unknown")

        feature_cols = [
            "year",
            "venue",
            "batting_team",
            "team1",
            "team2",
            "batter",
            "bowler"
        ]

        st.subheader("Select Batter vs Bowler Scenario")

        col1, col2 = st.columns(2)

        with col1:
            selected_batter_ml = st.selectbox(
                "Select Batter",
                sorted(matchup_base["batter"].dropna().unique()),
                key="ml_batter_vs_bowler_batter"
            )

            selected_venue_ml = st.selectbox(
                "Select Venue",
                sorted(matchup_base["venue"].dropna().unique()),
                key="ml_batter_vs_bowler_venue"
            )

            selected_team1_ml = st.selectbox(
                "Select Team 1",
                sorted(matchup_base["team1"].dropna().unique()),
                key="ml_batter_vs_bowler_team1"
            )

        with col2:
            selected_bowler_ml = st.selectbox(
                "Select Bowler",
                sorted(matchup_base["bowler"].dropna().unique()),
                key="ml_batter_vs_bowler_bowler"
            )

            selected_batting_team_ml = st.selectbox(
                "Select Batting Team",
                sorted(matchup_base["batting_team"].dropna().unique()),
                key="ml_batter_vs_bowler_batting_team"
            )

            selected_team2_ml = st.selectbox(
                "Select Team 2",
                sorted(matchup_base["team2"].dropna().unique()),
                key="ml_batter_vs_bowler_team2"
            )

        selected_year_ml = st.selectbox(
            "Select Prediction Year",
            sorted(matchup_base["year"].dropna().astype(int).unique()),
            key="ml_batter_vs_bowler_year"
        )

        input_df = pd.DataFrame([{
            "year": selected_year_ml,
            "venue": selected_venue_ml,
            "batting_team": selected_batting_team_ml,
            "team1": selected_team1_ml,
            "team2": selected_team2_ml,
            "batter": selected_batter_ml,
            "bowler": selected_bowler_ml
        }])

        if len(matchup_base) < 50:
            st.warning("Not enough historical matchup data to train a reliable model.")
        else:
            with st.spinner(f"Training {selected_model_family} models instantly..."):
                runs_model = build_rf_regression_model(
                    matchup_base,
                    feature_cols,
                    "runs_scored"
                )

                balls_model = build_rf_regression_model(
                    matchup_base,
                    feature_cols,
                    "balls_faced"
                )

                dismissed_model = None

                if matchup_base["dismissed"].nunique() > 1:
                    dismissed_model = build_rf_classification_model(
                        matchup_base,
                        feature_cols,
                        "dismissed"
                    )

                predicted_runs = max(0.0, float(runs_model.predict(input_df)[0]))
                predicted_balls = max(0.0, float(balls_model.predict(input_df)[0]))

                if dismissed_model is not None:
                    dismissal_probability = dismissed_model.predict_proba(input_df)[0][1]
                    dismissal_prediction = dismissed_model.predict(input_df)[0]
                else:
                    dismissal_probability = 0
                    dismissal_prediction = 0

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Predicted Runs", round(predicted_runs, 2))
            col2.metric("Predicted Balls Faced", round(predicted_balls, 2))
            col3.metric("Dismissal Probability", f"{round(dismissal_probability * 100, 2)}%")
            col4.metric(
                "Dismissal Prediction",
                "Out Chance" if dismissal_prediction == 1 else "Not Out Chance"
            )

            st.subheader("Historical Same Batter vs Bowler Data")

            historical_matchup = matchup_base[
                (matchup_base["batter"] == selected_batter_ml) &
                (matchup_base["bowler"] == selected_bowler_ml)
            ].sort_values("year", ascending=False)

            st.dataframe(historical_matchup, use_container_width=True)

            if not historical_matchup.empty:
                fig = px.bar(
                    historical_matchup,
                    x="year",
                    y="runs_scored",
                    color="venue",
                    title=f"Historical Runs: {selected_batter_ml} vs {selected_bowler_ml}",
                    text="runs_scored"
                )
                st.plotly_chart(fig, use_container_width=True)


# -------------------------
# ML - Match Winner Prediction
# -------------------------

elif page == "ML - Match Winner Prediction":
    st.header("ML - Match Winner Prediction")
    st.write(
        f"Selected model family: **{selected_model_family}**. The classification result is restricted to a selected two-team matchup. "
        "The displayed winning probabilities are restricted to the two selected teams only."
    )

    match_ml_df = filtered_matches_df.copy().dropna(subset=["winner"])

    for col in ["venue", "team1", "team2", "toss_winner", "toss_decision"]:
        match_ml_df[col] = match_ml_df[col].fillna("Unknown")

    if len(match_ml_df) < 20:
        st.warning("Not enough match data to train a reliable winner prediction model.")
    else:
        st.subheader("Select Match Scenario")

        teams = sorted(
            pd.concat([match_ml_df["team1"], match_ml_df["team2"]])
            .dropna()
            .unique()
        )
        venues = sorted(match_ml_df["venue"].dropna().unique())
        years = sorted(match_ml_df["year"].dropna().astype(int).unique())

        col1, col2 = st.columns(2)

        with col1:
            pred_year = st.selectbox("Select Year", years, key="winner_year")
            pred_team1 = st.selectbox("Select Team 1", teams, key="winner_team1")
            available_team2 = [team for team in teams if team != pred_team1]
            pred_toss_decision = st.selectbox(
                "Select Toss Decision",
                ["bat", "field"],
                key="winner_toss_decision"
            )

        with col2:
            pred_venue = st.selectbox("Select Venue", venues, key="winner_venue")
            pred_team2 = st.selectbox("Select Team 2", available_team2, key="winner_team2")
            pred_toss_winner = st.selectbox(
                "Select Toss Winner",
                [pred_team1, pred_team2],
                key="winner_toss"
            )

        feature_cols = [
            "year", "venue", "team1", "team2", "toss_winner", "toss_decision"
        ]

        input_df = pd.DataFrame([{
            "year": pred_year,
            "venue": pred_venue,
            "team1": pred_team1,
            "team2": pred_team2,
            "toss_winner": pred_toss_winner,
            "toss_decision": pred_toss_decision
        }])

        if st.button("Predict Match Winner", type="primary", key="basic_winner_button"):
            with st.spinner(f"Training {selected_model_family} winner prediction model..."):
                winner_model = build_rf_classification_model(
                    match_ml_df,
                    feature_cols,
                    "winner"
                )
                probabilities = winner_model.predict_proba(input_df)[0]
                classes = winner_model.classes_
                raw_probabilities = dict(zip(classes, probabilities))

                team1_raw = float(raw_probabilities.get(pred_team1, 0.0))
                team2_raw = float(raw_probabilities.get(pred_team2, 0.0))
                selected_total = team1_raw + team2_raw

                if selected_total == 0:
                    team1_probability = 0.5
                    team2_probability = 0.5
                else:
                    team1_probability = team1_raw / selected_total
                    team2_probability = team2_raw / selected_total

                prob_df = pd.DataFrame({
                    "Team": [pred_team1, pred_team2],
                    "Winning Probability (%)": [
                        round(team1_probability * 100, 2),
                        round(team2_probability * 100, 2)
                    ]
                }).sort_values("Winning Probability (%)", ascending=False)

                predicted_winner = prob_df.iloc[0]["Team"]

            st.success(f"Predicted Winner: {predicted_winner}")

            metric1, metric2 = st.columns(2)
            metric1.metric(f"{pred_team1} Win Probability", f"{team1_probability * 100:.2f}%")
            metric2.metric(f"{pred_team2} Win Probability", f"{team2_probability * 100:.2f}%")

            st.subheader("Winning Probability - Selected Teams Only")
            fig = px.bar(
                prob_df,
                x="Team",
                y="Winning Probability (%)",
                title=f"{pred_team1} vs {pred_team2}",
                text="Winning Probability (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            h2h = match_ml_df[
                (
                    ((match_ml_df["team1"] == pred_team1) & (match_ml_df["team2"] == pred_team2)) |
                    ((match_ml_df["team1"] == pred_team2) & (match_ml_df["team2"] == pred_team1))
                )
            ]
            st.subheader("Historical Head-to-Head Reference")
            if h2h.empty:
                st.info("No head-to-head matches are available in the selected year filter.")
            else:
                h2h_summary = (
                    h2h[h2h["winner"].isin([pred_team1, pred_team2])]
                    .groupby("winner")
                    .size()
                    .reindex([pred_team1, pred_team2], fill_value=0)
                    .reset_index(name="Wins")
                    .rename(columns={"winner": "Team"})
                )
                st.dataframe(h2h_summary, use_container_width=True, hide_index=True)

            st.caption(
                "Only the selected two teams are displayed. Their model probability mass is "
                "renormalized to a two-team contest."
            )


# -------------------------
# ML - Bowler Runs Prediction
# -------------------------

elif page == "ML - Bowler Runs Prediction":
    st.header("ML - Bowler Runs Prediction")
    st.write(f"Selected model family: **{selected_model_family}**. The regression model predicts bowler runs, balls and wickets.")

    bowler_ml_df = filtered_df.copy()

    bowler_match_df = (
        bowler_ml_df.groupby(
            [
                "match_id",
                "year",
                "venue",
                "bowler",
                "batting_team",
                "team1",
                "team2"
            ]
        )
        .agg(
            balls_bowled=("ball", "count"),
            runs_conceded=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            fours_conceded=("batter_runs", lambda x: (x == 4).sum()),
            sixes_conceded=("batter_runs", lambda x: (x == 6).sum())
        )
        .reset_index()
    )

    for col in ["venue", "batting_team", "team1", "team2", "bowler"]:
        bowler_match_df[col] = bowler_match_df[col].fillna("Unknown")

    if len(bowler_match_df) < 50:
        st.warning("Not enough bowler data to train a reliable model.")
    else:
        st.subheader("Select Bowler Scenario")

        col1, col2 = st.columns(2)

        with col1:
            selected_bowler_pred = st.selectbox(
                "Select Bowler",
                sorted(bowler_match_df["bowler"].dropna().unique()),
                key="bowler_pred_bowler"
            )

            selected_venue_pred = st.selectbox(
                "Select Venue",
                sorted(bowler_match_df["venue"].dropna().unique()),
                key="bowler_pred_venue"
            )

            selected_team1_pred = st.selectbox(
                "Select Team 1",
                sorted(bowler_match_df["team1"].dropna().unique()),
                key="bowler_pred_team1"
            )

        with col2:
            selected_batting_team_pred = st.selectbox(
                "Against Batting Team",
                sorted(bowler_match_df["batting_team"].dropna().unique()),
                key="bowler_pred_batting_team"
            )

            selected_team2_pred = st.selectbox(
                "Select Team 2",
                sorted(bowler_match_df["team2"].dropna().unique()),
                key="bowler_pred_team2"
            )

            selected_year_pred = st.selectbox(
                "Select Year",
                sorted(bowler_match_df["year"].dropna().astype(int).unique()),
                key="bowler_pred_year"
            )

        feature_cols = [
            "year",
            "venue",
            "bowler",
            "batting_team",
            "team1",
            "team2"
        ]

        input_df = pd.DataFrame([{
            "year": selected_year_pred,
            "venue": selected_venue_pred,
            "bowler": selected_bowler_pred,
            "batting_team": selected_batting_team_pred,
            "team1": selected_team1_pred,
            "team2": selected_team2_pred
        }])

        with st.spinner(f"Training {selected_model_family} bowler prediction models..."):
            runs_conceded_model = build_rf_regression_model(
                bowler_match_df,
                feature_cols,
                "runs_conceded"
            )

            balls_bowled_model = build_rf_regression_model(
                bowler_match_df,
                feature_cols,
                "balls_bowled"
            )

            wickets_model = build_rf_regression_model(
                bowler_match_df,
                feature_cols,
                "wickets"
            )

            predicted_runs_conceded = max(0.0, float(runs_conceded_model.predict(input_df)[0]))
            predicted_balls_bowled = max(0.0, float(balls_bowled_model.predict(input_df)[0]))
            predicted_wickets = max(0.0, float(wickets_model.predict(input_df)[0]))

        col1, col2, col3 = st.columns(3)

        col1.metric("Predicted Runs Conceded", round(predicted_runs_conceded, 2))
        col2.metric("Predicted Balls Bowled", round(predicted_balls_bowled, 2))
        col3.metric("Predicted Wickets", round(predicted_wickets, 2))

        st.subheader("Historical Bowler Data")

        historical_bowler = bowler_match_df[
            bowler_match_df["bowler"] == selected_bowler_pred
        ].sort_values("year", ascending=False)

        st.dataframe(historical_bowler, use_container_width=True)

        fig = px.bar(
            historical_bowler.head(30),
            x="year",
            y="runs_conceded",
            color="batting_team",
            title=f"Historical Runs Conceded by {selected_bowler_pred}",
            text="runs_conceded"
        )

        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# ML - Batter Runs Prediction
# -------------------------

elif page == "ML - Batter Runs Prediction":
    st.header("ML - Batter Runs Prediction")
    st.write(f"Selected model family: **{selected_model_family}**. The regression model predicts batter runs and balls; the classifier estimates out probability.")

    batter_ml_df = filtered_df.copy()

    batter_match_df = (
        batter_ml_df.groupby(
            [
                "match_id",
                "year",
                "venue",
                "batter",
                "batting_team",
                "team1",
                "team2"
            ]
        )
        .agg(
            balls_faced=("ball", "count"),
            runs_scored=("batter_runs", "sum"),
            fours=("batter_runs", lambda x: (x == 4).sum()),
            sixes=("batter_runs", lambda x: (x == 6).sum())
        )
        .reset_index()
    )

    out_df = (
        batter_ml_df[batter_ml_df["player_out"].notna()]
        .groupby(["match_id", "player_out"])
        .size()
        .reset_index(name="out_flag")
    )

    batter_match_df = batter_match_df.merge(
        out_df,
        left_on=["match_id", "batter"],
        right_on=["match_id", "player_out"],
        how="left"
    )

    batter_match_df["out_flag"] = batter_match_df["out_flag"].fillna(0)
    batter_match_df["out_flag"] = batter_match_df["out_flag"].apply(lambda x: 1 if x > 0 else 0)

    for col in ["venue", "batting_team", "team1", "team2", "batter"]:
        batter_match_df[col] = batter_match_df[col].fillna("Unknown")

    if len(batter_match_df) < 50:
        st.warning("Not enough batter data to train a reliable model.")
    else:
        st.subheader("Select Batter Scenario")

        col1, col2 = st.columns(2)

        with col1:
            selected_batter_pred = st.selectbox(
                "Select Batter",
                sorted(batter_match_df["batter"].dropna().unique()),
                key="batter_pred_batter"
            )

            selected_venue_pred = st.selectbox(
                "Select Venue",
                sorted(batter_match_df["venue"].dropna().unique()),
                key="batter_pred_venue"
            )

            selected_team1_pred = st.selectbox(
                "Select Team 1",
                sorted(batter_match_df["team1"].dropna().unique()),
                key="batter_pred_team1"
            )

        with col2:
            selected_batting_team_pred = st.selectbox(
                "Batting Team",
                sorted(batter_match_df["batting_team"].dropna().unique()),
                key="batter_pred_batting_team"
            )

            selected_team2_pred = st.selectbox(
                "Opponent / Team 2",
                sorted(batter_match_df["team2"].dropna().unique()),
                key="batter_pred_team2"
            )

            selected_year_pred = st.selectbox(
                "Select Year",
                sorted(batter_match_df["year"].dropna().astype(int).unique()),
                key="batter_pred_year"
            )

        feature_cols = [
            "year",
            "venue",
            "batter",
            "batting_team",
            "team1",
            "team2"
        ]

        input_df = pd.DataFrame([{
            "year": selected_year_pred,
            "venue": selected_venue_pred,
            "batter": selected_batter_pred,
            "batting_team": selected_batting_team_pred,
            "team1": selected_team1_pred,
            "team2": selected_team2_pred
        }])

        with st.spinner(f"Training {selected_model_family} batter prediction models..."):
            runs_model = build_rf_regression_model(
                batter_match_df,
                feature_cols,
                "runs_scored"
            )

            balls_model = build_rf_regression_model(
                batter_match_df,
                feature_cols,
                "balls_faced"
            )

            out_model = None

            if batter_match_df["out_flag"].nunique() > 1:
                out_model = build_rf_classification_model(
                    batter_match_df,
                    feature_cols,
                    "out_flag"
                )

            predicted_runs = runs_model.predict(input_df)[0]
            predicted_balls = balls_model.predict(input_df)[0]

            if out_model is not None:
                out_probability = out_model.predict_proba(input_df)[0][1]
            else:
                out_probability = 0

        col1, col2, col3 = st.columns(3)

        col1.metric("Predicted Runs", round(predicted_runs, 2))
        col2.metric("Predicted Balls Faced", round(predicted_balls, 2))
        col3.metric("Out Probability", f"{round(out_probability * 100, 2)}%")

        st.subheader("Historical Batter Data")

        historical_batter = batter_match_df[
            batter_match_df["batter"] == selected_batter_pred
        ].sort_values("year", ascending=False)

        st.dataframe(historical_batter, use_container_width=True)

        fig = px.bar(
            historical_batter.head(30),
            x="year",
            y="runs_scored",
            color="venue",
            title=f"Historical Runs by {selected_batter_pred}",
            text="runs_scored"
        )

        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# ML - Advanced Team Winner Prediction
# -------------------------

elif page == "ML - Advanced Team Winner Prediction":
    st.header("ML - Advanced Team Winner Prediction")
    st.write(
        f"Selected model family: **{selected_model_family}**. Predict the winning team using venue, toss and the historical batting/bowling "
        "strength of the selected playing XI."
    )

    with st.spinner("Preparing historical player-strength features..."):
        _, ratings, _, team_players = build_prediction_assets(full_df, matches_df)

    (
        advanced_team1,
        advanced_team2,
        advanced_venue,
        advanced_toss_winner,
        advanced_toss_decision,
        advanced_year,
        advanced_xi1,
        advanced_xi2
    ) = match_setup_inputs("advanced_winner", matches_df, team_players)

    if len(advanced_xi1) != 11 or len(advanced_xi2) != 11:
        st.warning("Please select exactly 11 players for both teams before prediction.")

    if st.button("Predict Advanced Match Winner", type="primary"):
        if len(advanced_xi1) != 11 or len(advanced_xi2) != 11:
            st.error("Both playing XIs must contain exactly 11 players.")
        else:
            with st.spinner(f"Training advanced {selected_model_family} winner model..."):
                advanced_training_df, ratings = build_advanced_match_training_data(
                    full_df,
                    matches_df
                )

                feature_cols = [
                    "year",
                    "venue",
                    "team_a",
                    "team_b",
                    "toss_winner",
                    "toss_decision",
                    "batting_strength_a",
                    "bowling_strength_a",
                    "batting_strength_b",
                    "bowling_strength_b"
                ]

                batting_strength_1, bowling_strength_1 = xi_strength(
                    advanced_xi1,
                    ratings
                )
                batting_strength_2, bowling_strength_2 = xi_strength(
                    advanced_xi2,
                    ratings
                )

                prediction_input = pd.DataFrame([{
                    "year": advanced_year,
                    "venue": advanced_venue,
                    "team_a": advanced_team1,
                    "team_b": advanced_team2,
                    "toss_winner": advanced_toss_winner,
                    "toss_decision": advanced_toss_decision,
                    "batting_strength_a": batting_strength_1,
                    "bowling_strength_a": bowling_strength_1,
                    "batting_strength_b": batting_strength_2,
                    "bowling_strength_b": bowling_strength_2
                }])

                advanced_model = build_rf_classification_model(
                    advanced_training_df,
                    feature_cols,
                    "team_a_win"
                )

                proba = advanced_model.predict_proba(prediction_input)[0]
                classes = list(advanced_model.classes_)
                probability_team1 = float(proba[classes.index(1)])
                probability_team2 = 1 - probability_team1

            predicted_team = (
                advanced_team1 if probability_team1 >= probability_team2 else advanced_team2
            )

            st.success(f"Predicted Winner: {predicted_team}")

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric(f"{advanced_team1} Win Probability", f"{probability_team1 * 100:.2f}%")
            metric2.metric(f"{advanced_team2} Win Probability", f"{probability_team2 * 100:.2f}%")
            metric3.metric("Prediction Confidence", f"{max(probability_team1, probability_team2) * 100:.2f}%")

            strength_df = pd.DataFrame({
                "Team": [advanced_team1, advanced_team2],
                "Batting Strength": [batting_strength_1, batting_strength_2],
                "Bowling Strength": [bowling_strength_1, bowling_strength_2]
            })
            st.subheader("Selected Playing XI Strength")
            st.dataframe(strength_df.round(2), use_container_width=True)

            historical_h2h = matches_df[
                (
                    ((matches_df["team1"] == advanced_team1) & (matches_df["team2"] == advanced_team2)) |
                    ((matches_df["team1"] == advanced_team2) & (matches_df["team2"] == advanced_team1))
                ) &
                (matches_df["winner"].notna())
            ]

            st.subheader("Historical Head-to-Head Summary")
            if historical_h2h.empty:
                st.info("No historical head-to-head matches were found in the available dataset.")
            else:
                h2h_summary = (
                    historical_h2h.groupby("winner")
                    .size()
                    .reset_index(name="Wins")
                    .rename(columns={"winner": "Team"})
                )
                st.dataframe(h2h_summary, use_container_width=True)


# -------------------------
# ML - Orange & Purple Cap Prediction
# -------------------------

elif page == "ML - Orange & Purple Cap Prediction":
    st.header("ML - Orange & Purple Cap Prediction")
    st.write(
        "Predict the season-ending leading run-scorer and wicket-taker using the actual "
        "recorded standings as the fixed baseline, plus only the remaining fixtures you select."
    )
    st.info(
        "The current season standings are never replaced by a new full-season prediction. "
        "Only players selected for future fixtures receive additional projected runs or wickets. "
        "Players not involved in selected remaining matches keep their recorded totals."
    )

    with st.spinner("Preparing season and player prediction assets..."):
        orange_stats, purple_stats = build_cap_season_tables(full_df)
        _, _, _, cap_team_players = build_prediction_assets(full_df, matches_df)

    latest_data_year = int(full_df["year"].dropna().max())
    all_teams = sorted(pd.concat([matches_df["team1"], matches_df["team2"]]).dropna().unique())
    all_venues = sorted(matches_df["venue"].dropna().unique())

    history_tab, forecast_tab = st.tabs(["Historical Actual Standings", "Fixture-Aware Cap Forecast"])

    with history_tab:
        season_years = sorted(orange_stats["year"].unique(), reverse=True)
        actual_year = st.selectbox("Select IPL Season", season_years, key="actual_cap_year")
        orange_actual = orange_stats[orange_stats["year"] == actual_year].sort_values(
            ["runs", "strike_rate"], ascending=[False, False]
        ).head(10)
        purple_actual = purple_stats[purple_stats["year"] == actual_year].sort_values(
            ["wickets", "economy"], ascending=[False, True]
        ).head(10)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Orange Cap Standings - {actual_year}")
            st.success(f'{orange_actual.iloc[0]["player"]} - {int(orange_actual.iloc[0]["runs"])} runs')
            st.dataframe(orange_actual.rename(columns={"player": "Player", "runs": "Runs", "matches": "Matches", "balls": "Balls", "strike_rate": "Strike Rate"}).round(2), use_container_width=True, hide_index=True)
        with col2:
            st.subheader(f"Purple Cap Standings - {actual_year}")
            st.success(f'{purple_actual.iloc[0]["player"]} - {int(purple_actual.iloc[0]["wickets"])} wickets')
            st.dataframe(purple_actual.rename(columns={"player": "Player", "wickets": "Wickets", "matches": "Matches", "balls": "Balls", "economy": "Economy"}).round(2), use_container_width=True, hide_index=True)

    with forecast_tab:
        recorded_year = st.selectbox(
            "Season containing completed / recorded matches",
            sorted(orange_stats["year"].unique(), reverse=True),
            index=0,
            key="fixture_cap_recorded_year",
            help="The runs and wickets already available in this season are treated as current totals."
        )

        st.subheader("Actual Standings Baseline Used for Projection")
        current_orange_baseline = orange_stats[orange_stats["year"] == recorded_year].sort_values(["runs", "strike_rate"], ascending=[False, False]).head(10)
        current_purple_baseline = purple_stats[purple_stats["year"] == recorded_year].sort_values(["wickets", "economy"], ascending=[False, True]).head(10)
        base1, base2 = st.columns(2)
        with base1:
            st.caption("Current Orange Cap top 10 before remaining fixtures")
            st.dataframe(current_orange_baseline[["player", "runs", "matches"]].rename(columns={"player": "Player", "runs": "Recorded Runs", "matches": "Matches"}), use_container_width=True, hide_index=True)
        with base2:
            st.caption("Current Purple Cap top 10 before remaining fixtures")
            st.dataframe(current_purple_baseline[["player", "wickets", "matches"]].rename(columns={"player": "Player", "wickets": "Recorded Wickets", "matches": "Matches"}), use_container_width=True, hide_index=True)

        st.subheader("Step 1: Select Remaining Fixtures")
        st.caption(
            "No CSV upload is required. Choose the number of remaining matches and select "
            "Team 1, Team 2 and Venue from dropdown menus for each fixture."
        )

        number_of_fixtures = st.number_input(
            "Number of remaining matches to include in the cap forecast",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
            key="remaining_fixture_count"
        )

        fixture_rows = []
        for fixture_no in range(int(number_of_fixtures)):
            with st.expander(f"Remaining Match {fixture_no + 1}", expanded=True):
                fixture_col1, fixture_col2, fixture_col3 = st.columns(3)
                with fixture_col1:
                    team1_index = fixture_no % len(all_teams)
                    fixture_team1 = st.selectbox(
                        "Team 1",
                        all_teams,
                        index=team1_index,
                        key=f"fixture_team1_{fixture_no}"
                    )
                available_opponents = [team for team in all_teams if team != fixture_team1]
                with fixture_col2:
                    fixture_team2 = st.selectbox(
                        "Team 2",
                        available_opponents,
                        index=fixture_no % len(available_opponents),
                        key=f"fixture_team2_{fixture_no}"
                    )
                with fixture_col3:
                    fixture_venue = st.selectbox(
                        "Venue",
                        all_venues,
                        index=fixture_no % len(all_venues),
                        key=f"fixture_venue_{fixture_no}"
                    )
                fixture_rows.append({
                    "Match": f"Remaining Match {fixture_no + 1}",
                    "Team 1": fixture_team1,
                    "Team 2": fixture_team2,
                    "Venue": fixture_venue
                })

        fixtures = pd.DataFrame(fixture_rows)
        st.subheader("Selected Remaining Fixture Summary")
        st.dataframe(fixtures, use_container_width=True, hide_index=True)

        if fixtures.empty:
            st.warning("Enter at least one valid future fixture to generate a fixture-aware cap prediction.")
        else:
            st.write(f"Valid remaining fixtures included in projection: **{len(fixtures)}**")
            st.subheader("Step 2: Select Projected Squad and Optional Impact Player")
            st.caption("Select the likely participating players from each team in your remaining fixture list. Use 11 or 12 players per team; the optional twelfth player can be marked as the Impact Player option.")
            participating_teams = sorted(set(fixtures["Team 1"]).union(set(fixtures["Team 2"])))
            squads = {}
            impacts = {}
            squad_valid = True
            for team in participating_teams:
                options = get_team_player_options(team, cap_team_players)
                default_pool = options[:12] if len(options) >= 12 else options
                with st.expander(f"{team} - Projected Squad / Player Pool", expanded=True):
                    selected_pool = st.multiselect(
                        f"Select players for {team}",
                        options,
                        default=default_pool,
                        key=f"cap_pool_{team}"
                    )
                    impact_choice = st.selectbox(
                        f"Optional Impact Player for {team}",
                        ["No Impact Player"] + selected_pool,
                        key=f"cap_impact_{team}"
                    )
                    if len(selected_pool) < 11:
                        st.warning(f"Select at least 11 players for {team}.")
                        squad_valid = False
                    squads[team] = selected_pool
                    impacts[team] = None if impact_choice == "No Impact Player" else impact_choice

            if st.button("Predict Orange Cap and Purple Cap from Remaining Fixtures", type="primary", key="predict_fixture_caps"):
                if not squad_valid:
                    st.error("Every participating team must have at least 11 selected players.")
                else:
                    with st.spinner("Calculating fixture-aware cap projections..."):
                        orange_projection, purple_projection = project_caps_from_future_fixtures(
                            full_df, int(recorded_year), fixtures, squads, impacts
                        )
                    if orange_projection.empty or purple_projection.empty:
                        st.warning("No eligible player projection could be generated from the selected fixtures and squads.")
                    else:
                        pred_col1, pred_col2 = st.columns(2)
                        with pred_col1:
                            st.subheader(f"Projected Orange Cap - {int(recorded_year)}")
                            st.success(
                                f'Predicted Winner: {orange_projection.iloc[0]["Player"]} '
                                f'({orange_projection.iloc[0]["Team"]}) - '
                                f'{int(orange_projection.iloc[0]["Projected Final Runs"])} runs'
                            )
                            orange_top = orange_projection.head(10)
                            st.dataframe(orange_top, use_container_width=True, hide_index=True)
                            st.plotly_chart(px.bar(orange_top, x="Player", y="Projected Final Runs", color="Team", text="Projected Final Runs", title="Fixture-Aware Orange Cap Projection"), use_container_width=True)
                        with pred_col2:
                            st.subheader(f"Projected Purple Cap - {int(recorded_year)}")
                            st.success(
                                f'Predicted Winner: {purple_projection.iloc[0]["Player"]} '
                                f'({purple_projection.iloc[0]["Team"]}) - '
                                f'{int(purple_projection.iloc[0]["Projected Final Wickets"])} wickets'
                            )
                            purple_top = purple_projection.head(10)
                            st.dataframe(purple_top, use_container_width=True, hide_index=True)
                            st.plotly_chart(px.bar(purple_top, x="Player", y="Projected Final Wickets", color="Team", text="Projected Final Wickets", title="Fixture-Aware Purple Cap Projection"), use_container_width=True)
                        dl1, dl2, dl3 = st.columns(3)
                        with dl1:
                            st.download_button("Download Orange Projection", orange_projection.to_csv(index=False).encode("utf-8"), file_name=f"fixture_orange_cap_projection_{int(recorded_year)}.csv", mime="text/csv")
                        with dl2:
                            st.download_button("Download Purple Projection", purple_projection.to_csv(index=False).encode("utf-8"), file_name=f"fixture_purple_cap_projection_{int(recorded_year)}.csv", mime="text/csv")
                        with dl3:
                            st.download_button("Download Remaining Fixtures", fixtures.to_csv(index=False).encode("utf-8"), file_name="remaining_ipl_fixtures_used.csv", mime="text/csv")
                        st.caption(
                            "Projection logic: actual recorded standings remain the baseline; only the selected squads in "
                            "selected remaining fixtures receive additional expected runs/wickets. Current-season per-match "
                            "form is used first, venue/opposition adjustments are deliberately limited, and an Impact Player "
                            "is treated as a conditional participant."
                        )


# -------------------------
# ML - Full Match Scorecard Simulation
# -------------------------

elif page == "ML - Full Match Scorecard Simulation":
    st.header("ML - Full Match Scorecard Simulation")
    st.write(
        "Generate a predicted IPL T20 scorecard delivery by delivery using historical "
        "batter, bowler and venue outcome distributions, with an optional Impact Player scenario."
    )
    st.info(
        "The Impact Player option is modelled as a user-selected substitution: the selected reserve "
        "replaces one named player in the active simulated XI before the innings are generated."
    )

    with st.spinner("Preparing player lists and match simulation distributions..."):
        _, ratings, _, team_players = build_prediction_assets(full_df, matches_df)
        distributions = build_simulation_distributions(full_df, matches_df)

    (
        simulation_team1,
        simulation_team2,
        simulation_venue,
        simulation_toss_winner,
        simulation_toss_decision,
        simulation_year,
        simulation_xi1,
        simulation_xi2
    ) = match_setup_inputs("scorecard", matches_df, team_players)

    st.subheader("Impact Player Scenario")
    use_impact = st.checkbox(
        "Enable Impact Player substitution for this predicted match",
        value=False,
        key="scorecard_enable_impact"
    )
    active_xi1 = list(simulation_xi1)
    active_xi2 = list(simulation_xi2)
    impact_text1 = "No Impact Player substitution"
    impact_text2 = "No Impact Player substitution"

    if use_impact and len(simulation_xi1) == 11 and len(simulation_xi2) == 11:
        ip_col1, ip_col2 = st.columns(2)
        with ip_col1:
            reserve_options1 = [p for p in get_team_player_options(simulation_team1, team_players) if p not in simulation_xi1]
            impact1 = st.selectbox(f"{simulation_team1} Impact Player", ["No Impact Player"] + reserve_options1, key="scorecard_impact1")
            replace1 = st.selectbox(f"{simulation_team1} player replaced", simulation_xi1, key="scorecard_replace1") if impact1 != "No Impact Player" else None
            active_xi1 = apply_impact_replacement(simulation_xi1, impact1, replace1)
            if impact1 != "No Impact Player":
                impact_text1 = f"{impact1} replaces {replace1}"
        with ip_col2:
            reserve_options2 = [p for p in get_team_player_options(simulation_team2, team_players) if p not in simulation_xi2]
            impact2 = st.selectbox(f"{simulation_team2} Impact Player", ["No Impact Player"] + reserve_options2, key="scorecard_impact2")
            replace2 = st.selectbox(f"{simulation_team2} player replaced", simulation_xi2, key="scorecard_replace2") if impact2 != "No Impact Player" else None
            active_xi2 = apply_impact_replacement(simulation_xi2, impact2, replace2)
            if impact2 != "No Impact Player":
                impact_text2 = f"{impact2} replaces {replace2}"
        st.write(f"**{simulation_team1}:** {impact_text1}  \n**{simulation_team2}:** {impact_text2}")
    elif use_impact:
        st.warning("First select exactly 11 starting players for both teams to enable Impact Player replacement.")

    bowl_col1, bowl_col2 = st.columns(2)
    with bowl_col1:
        simulation_bowlers1 = st.multiselect(
            f"Select bowlers for {simulation_team1}",
            active_xi1,
            default=get_default_bowlers(simulation_team1, active_xi1, team_players),
            key="scorecard_bowlers1_impact",
            help="Select at least five active bowlers. Each bowler is limited to four overs."
        )
    with bowl_col2:
        simulation_bowlers2 = st.multiselect(
            f"Select bowlers for {simulation_team2}",
            active_xi2,
            default=get_default_bowlers(simulation_team2, active_xi2, team_players),
            key="scorecard_bowlers2_impact",
            help="Select at least five active bowlers. Each bowler is limited to four overs."
        )

    valid_scorecard_input = (
        len(active_xi1) == 11 and len(active_xi2) == 11 and
        len(set(active_xi1)) == 11 and len(set(active_xi2)) == 11 and
        len(simulation_bowlers1) >= 5 and len(simulation_bowlers2) >= 5
    )
    if not valid_scorecard_input:
        st.warning("Select exactly 11 unique active players and at least 5 active bowlers for each team.")

    if st.button("Generate Predicted Full Scorecard", type="primary", key="generate_scorecard_with_impact"):
        if not valid_scorecard_input:
            st.error("Cannot simulate the match. Check active XIs and bowlers.")
        else:
            if simulation_toss_decision == "bat":
                batting_first = simulation_toss_winner
            else:
                batting_first = simulation_team2 if simulation_toss_winner == simulation_team1 else simulation_team1
            bowling_first = simulation_team2 if batting_first == simulation_team1 else simulation_team1
            first_xi = active_xi1 if batting_first == simulation_team1 else active_xi2
            second_xi = active_xi2 if batting_first == simulation_team1 else active_xi1
            first_innings_bowlers = simulation_bowlers2 if batting_first == simulation_team1 else simulation_bowlers1
            second_innings_bowlers = simulation_bowlers1 if batting_first == simulation_team1 else simulation_bowlers2

            stable_seed_text = "|".join([
                simulation_team1, simulation_team2, simulation_venue, simulation_toss_winner,
                simulation_toss_decision, str(simulation_year), ",".join(active_xi1),
                ",".join(active_xi2), ",".join(simulation_bowlers1), ",".join(simulation_bowlers2),
                impact_text1, impact_text2
            ])
            stable_seed = int(hashlib.md5(stable_seed_text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(stable_seed)

            with st.spinner("Simulating 20-over match scorecard including Impact Player scenario..."):
                first_innings = simulate_innings(
                    batting_team=batting_first, bowling_team=bowling_first, batting_xi=first_xi,
                    bowling_xi=second_xi, bowlers=first_innings_bowlers, venue=simulation_venue,
                    innings_number=1, distributions=distributions, rng=rng
                )
                target = first_innings["score"] + 1
                second_innings = simulate_innings(
                    batting_team=bowling_first, bowling_team=batting_first, batting_xi=second_xi,
                    bowling_xi=first_xi, bowlers=second_innings_bowlers, venue=simulation_venue,
                    innings_number=2, distributions=distributions, rng=rng, target=target
                )

            if second_innings["score"] >= target:
                predicted_winner = bowling_first
                result_text = f"{predicted_winner} predicted to win by {10 - second_innings['wickets']} wicket(s)."
            elif first_innings["score"] > second_innings["score"]:
                predicted_winner = batting_first
                result_text = f"{predicted_winner} predicted to win by {first_innings['score'] - second_innings['score']} run(s)."
            else:
                predicted_winner = "Tie"
                result_text = "The predicted match ends in a tie."

            st.subheader("Predicted Match Result")
            st.success(result_text)
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            summary_col1.metric("Venue", simulation_venue)
            summary_col2.metric("Toss Winner", simulation_toss_winner)
            summary_col3.metric(batting_first, f'{first_innings["score"]}/{first_innings["wickets"]} ({first_innings["overs"]})')
            summary_col4.metric(bowling_first, f'{second_innings["score"]}/{second_innings["wickets"]} ({second_innings["overs"]})')
            st.write(
                f"**Toss Decision:** {simulation_toss_winner} elected to {simulation_toss_decision}.  \n"
                f"**Batting First:** {batting_first}  \n**Target:** {target}  \n"
                f"**{simulation_team1} Impact Scenario:** {impact_text1}  \n"
                f"**{simulation_team2} Impact Scenario:** {impact_text2}"
            )

            innings1_tab, innings2_tab, progress_tab = st.tabs(["First Innings Scorecard", "Second Innings Scorecard", "Over-by-Over Progress"])
            with innings1_tab:
                st.subheader(f'{first_innings["batting_team"]}: {first_innings["score"]}/{first_innings["wickets"]} in {first_innings["overs"]} overs')
                st.markdown("**Batting Scorecard**")
                st.dataframe(first_innings["batting"], use_container_width=True)
                st.markdown("**Bowling Scorecard**")
                st.dataframe(first_innings["bowling"], use_container_width=True)
                st.markdown("**Fall of Wickets**")
                st.write(", ".join(first_innings["fall_of_wickets"]) if first_innings["fall_of_wickets"] else "No wickets")
            with innings2_tab:
                st.subheader(f'{second_innings["batting_team"]}: {second_innings["score"]}/{second_innings["wickets"]} in {second_innings["overs"]} overs')
                st.markdown("**Batting Scorecard**")
                st.dataframe(second_innings["batting"], use_container_width=True)
                st.markdown("**Bowling Scorecard**")
                st.dataframe(second_innings["bowling"], use_container_width=True)
                st.markdown("**Fall of Wickets**")
                st.write(", ".join(second_innings["fall_of_wickets"]) if second_innings["fall_of_wickets"] else "No wickets")
            with progress_tab:
                full_ball_log = pd.concat([first_innings["ball_log"], second_innings["ball_log"]], ignore_index=True)
                if not full_ball_log.empty:
                    full_ball_log["Ball Number"] = full_ball_log.groupby("Innings").cumcount() + 1
                    st.plotly_chart(px.line(full_ball_log, x="Ball Number", y="Score", color="Batting Team", title="Predicted Innings Score Progression"), use_container_width=True)
                    st.dataframe(full_ball_log, use_container_width=True)

            combined_batting_csv = pd.concat([
                first_innings["batting"].assign(Innings=1, Team=first_innings["batting_team"]),
                second_innings["batting"].assign(Innings=2, Team=second_innings["batting_team"])
            ], ignore_index=True)
            combined_bowling_csv = pd.concat([
                first_innings["bowling"].assign(Innings=1, Team=first_innings["bowling_team"]),
                second_innings["bowling"].assign(Innings=2, Team=second_innings["bowling_team"])
            ], ignore_index=True)
            all_ball_log = pd.concat([first_innings["ball_log"], second_innings["ball_log"]], ignore_index=True)
            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button("Download Predicted Batting Scorecard", combined_batting_csv.to_csv(index=False).encode("utf-8"), file_name="predicted_ipl_batting_scorecard.csv", mime="text/csv")
            with d2:
                st.download_button("Download Predicted Bowling Scorecard", combined_bowling_csv.to_csv(index=False).encode("utf-8"), file_name="predicted_ipl_bowling_scorecard.csv", mime="text/csv")
            with d3:
                st.download_button("Download Predicted Ball-by-Ball Data", all_ball_log.to_csv(index=False).encode("utf-8"), file_name="predicted_ipl_ball_by_ball.csv", mime="text/csv")
            st.caption("Simulation note: outcomes are generated from historical batter, bowler and venue delivery distributions. The Impact Player represents the replacement scenario you selected.")


# -------------------------
# Download Filtered Data
# -------------------------

elif page == "Download Filtered Data":
    st.header("Download Filtered Data")

    if year_filter_type == "Single Year":
        st.write(f"Current filter: IPL data for {selected_year}")
    else:
        st.write(
            f"Current filter: IPL data from {selected_year_range[0]} to {selected_year_range[1]}"
        )

    st.subheader("Filtered Ball-by-Ball Data")

    row_limit = st.number_input(
        "Number of rows to display",
        min_value=100,
        max_value=max(100, len(filtered_df)),
        value=min(1000, len(filtered_df)),
        step=100
    )

    st.dataframe(filtered_df.head(row_limit), use_container_width=True)

    ball_csv = filtered_df.to_csv(index=False).encode("utf-8")

    ball_file_name = (
        f"ipl_ball_by_ball_{selected_year}.csv"
        if year_filter_type == "Single Year"
        else f"ipl_ball_by_ball_{selected_year_range[0]}_{selected_year_range[1]}.csv"
    )

    st.download_button(
        label="Download Filtered Ball-by-Ball CSV",
        data=ball_csv,
        file_name=ball_file_name,
        mime="text/csv"
    )

    st.subheader("Filtered Matches Data")

    st.dataframe(filtered_matches_df, use_container_width=True)

    match_csv = filtered_matches_df.to_csv(index=False).encode("utf-8")

    match_file_name = (
        f"ipl_matches_{selected_year}.csv"
        if year_filter_type == "Single Year"
        else f"ipl_matches_{selected_year_range[0]}_{selected_year_range[1]}.csv"
    )

    st.download_button(
        label="Download Filtered Matches CSV",
        data=match_csv,
        file_name=match_file_name,
        mime="text/csv"
    )