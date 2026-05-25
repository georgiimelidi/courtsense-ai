import pandas as pd

from predict import predict_match
from predict import get_surface_elo


ROUND_NAMES = [
    "R128",
    "R64",
    "R32",
    "R16",
    "QF",
    "SF",
    "F",
]


def get_player_row(player_stats, player_name):
    row = player_stats[player_stats["player_name"] == player_name]

    if row.empty:
        return None

    return row.iloc[0]


def predict_simple_match(player_stats, player_a_name, player_b_name, surface="Clay"):
    player_a = get_player_row(player_stats, player_a_name)
    player_b = get_player_row(player_stats, player_b_name)

    if player_a is None or player_b is None:
        return None

    player_a_surface_elo = get_surface_elo(player_a, surface)
    player_b_surface_elo = get_surface_elo(player_b, surface)

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
        best_of=5,
        surface=surface,
        player_a_recent_win_rate=surface_form_a,
        player_b_recent_win_rate=surface_form_b,
    )

    return result


def simulate_tournament(draw, player_stats, surface="Clay"):
    current_matches = draw[["player_a", "player_b"]].copy()

    all_results = []

    round_index = 0

    while len(current_matches) >= 1:
        round_name = ROUND_NAMES[round_index]

        winners = []

        for _, match in current_matches.iterrows():
            player_a = match["player_a"]
            player_b = match["player_b"]

            result = predict_simple_match(
                player_stats,
                player_a,
                player_b,
                surface=surface,
            )

            if result is None:
                winner = player_a
                player_a_prob = None
                player_b_prob = None
            else:
                winner = result["predicted_winner"]
                player_a_prob = result["player_a_win_probability"]
                player_b_prob = result["player_b_win_probability"]

            winners.append(winner)

            all_results.append(
                {
                    "round": round_name,
                    "player_a": player_a,
                    "player_b": player_b,
                    "predicted_winner": winner,
                    "player_a_win_probability": player_a_prob,
                    "player_b_win_probability": player_b_prob,
                }
            )

        if len(winners) == 1:
            break

        next_matches = []

        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                next_matches.append(
                    {
                        "player_a": winners[i],
                        "player_b": winners[i + 1],
                    }
                )

        current_matches = pd.DataFrame(next_matches)
        round_index += 1

    return pd.DataFrame(all_results)