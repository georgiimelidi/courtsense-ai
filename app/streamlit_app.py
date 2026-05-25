import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from predict import predict_match, make_match_features
from retrieval import find_similar_matches
from head_to_head import get_head_to_head
from draw_loader import load_roland_garros_draw
from tournament_simulator import simulate_tournament
from explainability import get_feature_importance
from agents import (
    ranking_agent,
    elo_agent,
    surface_agent,
    form_agent,
    fatigue_agent,
    age_agent,
    h2h_agent_from_summary,
    matchup_agent,
    similarity_rag_agent,
    news_context_agent,
    upset_risk_agent,
    judge_agent,
)


PLAYER_STATS_PATH = ROOT_DIR / "data" / "player_stats.csv"


def get_surface_elo(player, surface):
    if surface == "Clay":
        return player["clay_elo"]
    if surface == "Hard":
        return player["hard_elo"]
    if surface == "Grass":
        return player["grass_elo"]
    return player["overall_elo"]


def get_default_index(player_names, name, fallback):
    if name in player_names:
        return player_names.index(name)
    return fallback


def add_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --navy: #18223C;
            --blue: #3B82F6;
            --sky: #DBEAFE;
            --green: #2E7D5B;
            --mint: #DFF7EC;
            --gold: #C99A2E;
            --cream: #FAF7F0;
            --gray: #F3F4F6;
            --text: #111827;
        }

        .stApp {
            background: linear-gradient(180deg, #FAF7F0 0%, #FFFFFF 45%, #F8FAFC 100%);
            color: var(--text);
        }

        .main .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }

        h1, h2, h3 {
            color: var(--navy);
            letter-spacing: -0.02em;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #18223C 0%, #25345C 55%, #3B82F6 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(24, 34, 60, 0.18);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            color: white;
            font-size: 2.7rem;
            margin-bottom: 0.3rem;
        }

        .hero p {
            color: #E5E7EB;
            font-size: 1.05rem;
            margin-bottom: 0;
        }

        .section-card {
            padding: 1.3rem 1.5rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(24, 34, 60, 0.08);
            box-shadow: 0 12px 30px rgba(24, 34, 60, 0.08);
            margin-bottom: 1.2rem;
        }

        .small-note {
            color: #6B7280;
            font-size: 0.95rem;
        }

        .champion-card {
            padding: 1.2rem 1.4rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #FFF7D6 0%, #FDECC8 100%);
            border: 1px solid rgba(201, 154, 46, 0.25);
            color: #654B08;
            font-size: 1.15rem;
            font-weight: 700;
            margin: 1rem 0;
        }

        .prediction-card {
            padding: 1.3rem 1.5rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #EEF6FF 0%, #FFFFFF 100%);
            border: 1px solid rgba(59, 130, 246, 0.16);
            box-shadow: 0 12px 30px rgba(59, 130, 246, 0.10);
            margin-bottom: 1rem;
        }

        .footer-card {
            padding: 1.4rem 1.6rem;
            border-radius: 24px;
            background: #18223C;
            color: white;
            margin-top: 2rem;
        }

        .footer-card h3 {
            color: white;
            margin-bottom: 0.3rem;
        }

        .footer-card p {
            color: #E5E7EB;
            margin-bottom: 0.2rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid rgba(24, 34, 60, 0.08);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 8px 22px rgba(24, 34, 60, 0.06);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }

        .stButton > button {
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #18223C 0%, #3B82F6 100%);
            color: white;
            padding: 0.65rem 1.25rem;
            font-weight: 700;
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.25);
        }

        .stButton > button:hover {
            color: white;
            filter: brightness(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="CourtSense AI",
    page_icon="🎾",
    layout="wide",
)

add_custom_css()

st.markdown(
    """
    <div class="hero">
        <h1>CourtSense AI</h1>
        <p>
        Machine learning system for tennis match prediction and tournament simulation,
        combining Elo ratings, historical retrieval, and multi-agent analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

player_stats = pd.read_csv(PLAYER_STATS_PATH)

ranking_filter = st.sidebar.radio(
    "Player list",
    ["Top 30", "Top 50", "All players"],
    index=0,
)

if ranking_filter == "Top 30":
    filtered_stats = player_stats[player_stats["rank"] <= 30]
elif ranking_filter == "Top 50":
    filtered_stats = player_stats[player_stats["rank"] <= 50]
else:
    filtered_stats = player_stats

player_names = sorted(filtered_stats["player_name"].unique().tolist())


# -------------------------------------------------------------------
# Roland-Garros simulation
# -------------------------------------------------------------------

st.markdown("## Roland-Garros Simulation")

st.markdown(
    """
    <div class="section-card">
        <b>Goal.</b> Simulate the men's singles draw using the CourtSense prediction model.
        The official draw is used as input, but the visible output is a clean projected tournament path.
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("Simulate Roland-Garros"):
    try:
        draw = load_roland_garros_draw()

        simulation = simulate_tournament(
            draw=draw,
            player_stats=player_stats,
            surface="Clay",
        )

        champion = simulation.iloc[-1]["predicted_winner"]

        st.markdown(
            f"""
            <div class="champion-card">
                Predicted champion: {champion}
            </div>
            """,
            unsafe_allow_html=True,
        )

        round_order = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]

        for round_name in round_order:
            round_df = simulation[simulation["round"] == round_name].copy()

            if round_df.empty:
                continue

            with st.expander(f"{round_name} projected matches", expanded=(round_name in ["SF", "F"])):
                display_round = round_df[
                    [
                        "player_a",
                        "player_b",
                        "predicted_winner",
                        "player_a_win_probability",
                        "player_b_win_probability",
                    ]
                ].copy()

                display_round["player_a_win_probability"] = display_round[
                    "player_a_win_probability"
                ].apply(lambda x: "" if pd.isna(x) else f"{x:.1%}")

                display_round["player_b_win_probability"] = display_round[
                    "player_b_win_probability"
                ].apply(lambda x: "" if pd.isna(x) else f"{x:.1%}")

                st.dataframe(
                    display_round,
                    use_container_width=True,
                    hide_index=True,
                )

    except Exception as e:
        st.error(f"Could not simulate tournament: {e}")


# -------------------------------------------------------------------
# Match predictor
# -------------------------------------------------------------------

st.markdown("---")
st.markdown("## Match Predictor")

default_a = get_default_index(player_names, "Jannik Sinner", 0)
default_b = get_default_index(
    player_names,
    "Alexander Zverev",
    min(1, len(player_names) - 1),
)

col1, col2 = st.columns(2)

with col1:
    player_a_name = st.selectbox(
        "Player A",
        player_names,
        index=default_a,
    )

with col2:
    player_b_name = st.selectbox(
        "Player B",
        player_names,
        index=default_b,
    )

control_col1, control_col2 = st.columns(2)

with control_col1:
    surface = st.selectbox("Surface", ["Clay", "Hard", "Grass"], index=0)

with control_col2:
    best_of = st.selectbox("Best of", [3, 5], index=1)

player_a = player_stats[player_stats["player_name"] == player_a_name].iloc[0]
player_b = player_stats[player_stats["player_name"] == player_b_name].iloc[0]

player_a_surface_elo = get_surface_elo(player_a, surface)
player_b_surface_elo = get_surface_elo(player_b, surface)

with st.expander("Player information", expanded=False):
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown(f"### {player_a_name}")
        st.write(f"Rank: {int(player_a['rank'])}")
        st.write(f"Points: {int(player_a['rank_points'])}")
        st.write(f"Age: {int(player_a['age'])}")
        st.write(f"Recent form: {player_a['recent_win_rate']:.2f}")
        st.write(f"Clay form: {player_a['recent_clay_win_rate']:.2f}")
        st.write(f"Overall Elo: {player_a['overall_elo']:.0f}")
        st.write(f"{surface} Elo: {player_a_surface_elo:.0f}")
        st.write(f"Serve strength: {player_a.get('serve_strength', 0.5):.2f}")
        st.write(f"Return strength: {player_a.get('return_strength', 0.5):.2f}")
        st.write(f"Matches in last {int(player_a.get('fatigue_window_days', 14))} days: {int(player_a.get('recent_match_count', 0))}")
        st.write(f"Historical matches in dataset: {int(player_a['matches_played'])}")

    with info_col2:
        st.markdown(f"### {player_b_name}")
        st.write(f"Rank: {int(player_b['rank'])}")
        st.write(f"Points: {int(player_b['rank_points'])}")
        st.write(f"Age: {int(player_b['age'])}")
        st.write(f"Recent form: {player_b['recent_win_rate']:.2f}")
        st.write(f"Clay form: {player_b['recent_clay_win_rate']:.2f}")
        st.write(f"Overall Elo: {player_b['overall_elo']:.0f}")
        st.write(f"{surface} Elo: {player_b_surface_elo:.0f}")
        st.write(f"Serve strength: {player_b.get('serve_strength', 0.5):.2f}")
        st.write(f"Return strength: {player_b.get('return_strength', 0.5):.2f}")
        st.write(f"Matches in last {int(player_b.get('fatigue_window_days', 14))} days: {int(player_b.get('recent_match_count', 0))}")
        st.write(f"Historical matches in dataset: {int(player_b['matches_played'])}")


if st.button("Predict match"):

    surface_form_a = (
        player_a["recent_clay_win_rate"]
        if surface == "Clay"
        else player_a["recent_win_rate"]
    )

    surface_form_b = (
        player_b["recent_clay_win_rate"]
        if surface == "Clay"
        else player_b["recent_win_rate"]
    )

    result = predict_match(
        player_a_name=player_a_name,
        player_b_name=player_b_name,
        player_a_rank=player_a["rank"],
        player_b_rank=player_b["rank"],
        player_a_rank_points=player_a["rank_points"],
        player_b_rank_points=player_b["rank_points"],
        player_a_age=player_a["age"],
        player_b_age=player_b["age"],
        player_a_overall_elo=player_a["overall_elo"],
        player_b_overall_elo=player_b["overall_elo"],
        player_a_surface_elo=player_a_surface_elo,
        player_b_surface_elo=player_b_surface_elo,
        best_of=best_of,
        surface=surface,
        player_a_recent_win_rate=surface_form_a,
        player_b_recent_win_rate=surface_form_b,
    )

    query_features = make_match_features(
        player_a_rank=player_a["rank"],
        player_b_rank=player_b["rank"],
        player_a_rank_points=player_a["rank_points"],
        player_b_rank_points=player_b["rank_points"],
        player_a_age=player_a["age"],
        player_b_age=player_b["age"],
        player_a_overall_elo=player_a["overall_elo"],
        player_b_overall_elo=player_b["overall_elo"],
        player_a_surface_elo=player_a_surface_elo,
        player_b_surface_elo=player_b_surface_elo,
        best_of=best_of,
        surface=surface,
        player_a_recent_win_rate=surface_form_a,
        player_b_recent_win_rate=surface_form_b,
    )

    similar_matches = find_similar_matches(query_features, top_k=5)
    h2h = get_head_to_head(player_a_name, player_b_name)

    st.markdown(
        f"""
        <div class="prediction-card">
            <h2>Prediction</h2>
            <p><b>Predicted winner:</b> {result["predicted_winner"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prob_col1, prob_col2, conf_col = st.columns(3)

    with prob_col1:
        st.metric(
            f"{player_a_name}",
            f"{result['player_a_win_probability']:.1%}",
        )

    with prob_col2:
        st.metric(
            f"{player_b_name}",
            f"{result['player_b_win_probability']:.1%}",
        )

    winner_prob = result["winner_probability"]

    if winner_prob > 0.70:
        confidence = "High"
    elif winner_prob > 0.58:
        confidence = "Medium"
    else:
        confidence = "Low"

    with conf_col:
        st.metric("Confidence", confidence)

    # ---------------------------------------------------------------
    # Agent analysis
    # ---------------------------------------------------------------

    st.markdown("## Contextual Agent Analysis")
    st.caption(
        "The XGBoost model produces the probability. Agents add contextual evidence "
        "from head-to-head history, retrieval, news, fatigue, matchup."
    )

    agent_outputs = [
        h2h_agent_from_summary(player_a_name, player_b_name, h2h, surface),
        matchup_agent(player_a, player_b),
        fatigue_agent(player_a, player_b),
        similarity_rag_agent(similar_matches, player_a_name, player_b_name),
        news_context_agent(player_a_name, player_b_name),
        upset_risk_agent(result, player_a, player_b),
    ]

    final_judgement = judge_agent(result, agent_outputs)

    agent_table = pd.DataFrame(agent_outputs)

    player_a_votes = (agent_table["favored_player"] == player_a_name).sum()
    player_b_votes = (agent_table["favored_player"] == player_b_name).sum()
    neutral_votes = agent_table["favored_player"].isna().sum()

    vote_col1, vote_col2, vote_col3 = st.columns(3)

    with vote_col1:
        st.metric(player_a_name, f"{player_a_votes} votes")

    with vote_col2:
        st.metric(player_b_name, f"{player_b_votes} votes")

    with vote_col3:
        st.metric("Neutral", f"{neutral_votes} agents")

    def color_favor(value):
        if value == player_a_name:
            return "background-color: #DBEAFE; color: #0B3D91;"
        if value == player_b_name:
            return "background-color: #DFF7EC; color: #166534;"
        return "background-color: #F3F4F6; color: #4B5563;"

    display_agent_table = (
        agent_table[
            ["agent", "signal", "favored_player", "evidence", "message"]
        ]
        .fillna("Neutral")
    )

    styled_agent_table = display_agent_table.style.applymap(
        color_favor,
        subset=["favored_player"],
    )

    st.dataframe(
        styled_agent_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## Final judgement")
    st.info(final_judgement["message"])

    # ---------------------------------------------------------------
    # Hidden evidence
    # ---------------------------------------------------------------

    with st.expander("News evidence", expanded=False):
        from news_context import analyze_news

        news = analyze_news(player_a_name, player_b_name)

        col_news_a, col_news_b = st.columns(2)

        with col_news_a:
            st.markdown(f"### {player_a_name}")
            for item in news["player_a"]["headlines"][:5]:
                st.write(f"- {item['title']}")

        with col_news_b:
            st.markdown(f"### {player_b_name}")
            for item in news["player_b"]["headlines"][:5]:
                st.write(f"- {item['title']}")

    with st.expander("Similar historical matches", expanded=False):
        display_matches = similar_matches[
            [
                "winner_name",
                "loser_name",
                "surface",
                "tourney_name",
                "tourney_date",
                "score",
                "similarity",
            ]
        ].copy()

        display_matches["similarity"] = display_matches["similarity"].round(3)
        st.dataframe(display_matches, use_container_width=True, hide_index=True)

    with st.expander("Head-to-head matches", expanded=False):
        h2h_matches = h2h["matches"]

        if h2h_matches.empty:
            st.write("No head-to-head matches found in the historical dataset.")
        else:
            st.dataframe(
                h2h_matches[
                    [
                        "tourney_date",
                        "tourney_name",
                        "surface",
                        "winner_name",
                        "loser_name",
                        "score",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Model explainability", expanded=False):
        importance_df = get_feature_importance()

        st.bar_chart(
            importance_df.set_index("feature")["importance"]
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("System notes", expanded=False):
        st.write(
            """
            Current version uses:
            - current ranking and ranking points,
            - player age,
            - recent form,
            - recent clay form,
            - overall Elo,
            - surface-specific Elo,
            - serve/return profile,
            - recent match load,
            - historical similar-match retrieval,
            - head-to-head analysis,
            - lightweight news context retrieval,
            - agent-based explanation.
            """
        )


# -------------------------------------------------------------------
# Credentials
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="
        margin-top: 2rem;
        padding-top: 1rem;
        text-align: center;
        color: #6B7280;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    ">
        Built by <b style="color:#18223C;">Georgii Melidi</b> · 2026
    </div>
    """,
    unsafe_allow_html=True,
)