import joblib
import pandas as pd

from features import FEATURE_COLUMNS


MODEL_PATH = "model.pkl"


def get_surface_elo(player, surface):
    if surface == "Clay":
        return player["clay_elo"]
    if surface == "Hard":
        return player["hard_elo"]
    if surface == "Grass":
        return player["grass_elo"]
    return player["overall_elo"]


def make_match_features(
    player_a_rank,
    player_b_rank,
    player_a_rank_points,
    player_b_rank_points,
    player_a_age,
    player_b_age,
    player_a_overall_elo,
    player_b_overall_elo,
    player_a_surface_elo,
    player_b_surface_elo,
    best_of=5,
    surface="Clay",
    player_a_recent_win_rate=0.5,
    player_b_recent_win_rate=0.5,
):
    return pd.DataFrame(
        [
            {
                "rank_diff": player_a_rank - player_b_rank,
                "rank_points_diff": player_a_rank_points - player_b_rank_points,
                "age_diff": player_a_age - player_b_age,
                "best_of": best_of,
                "is_clay": int(surface == "Clay"),
                "is_hard": int(surface == "Hard"),
                "is_grass": int(surface == "Grass"),
                "recent_win_rate_diff": (
                    player_a_recent_win_rate - player_b_recent_win_rate
                ),
                "overall_elo_diff": player_a_overall_elo - player_b_overall_elo,
                "surface_elo_diff": player_a_surface_elo - player_b_surface_elo,
            }
        ]
    )[FEATURE_COLUMNS]


def predict_match(
    player_a_name,
    player_b_name,
    player_a_rank,
    player_b_rank,
    player_a_rank_points,
    player_b_rank_points,
    player_a_age,
    player_b_age,
    player_a_overall_elo,
    player_b_overall_elo,
    player_a_surface_elo,
    player_b_surface_elo,
    best_of=5,
    surface="Clay",
    player_a_recent_win_rate=0.5,
    player_b_recent_win_rate=0.5,
):
    model = joblib.load(MODEL_PATH)

    X = make_match_features(
        player_a_rank=player_a_rank,
        player_b_rank=player_b_rank,
        player_a_rank_points=player_a_rank_points,
        player_b_rank_points=player_b_rank_points,
        player_a_age=player_a_age,
        player_b_age=player_b_age,
        player_a_overall_elo=player_a_overall_elo,
        player_b_overall_elo=player_b_overall_elo,
        player_a_surface_elo=player_a_surface_elo,
        player_b_surface_elo=player_b_surface_elo,
        best_of=best_of,
        surface=surface,
        player_a_recent_win_rate=player_a_recent_win_rate,
        player_b_recent_win_rate=player_b_recent_win_rate,
    )

    player_a_win_prob = model.predict_proba(X)[0, 1]
    player_b_win_prob = 1 - player_a_win_prob

    if player_a_win_prob >= player_b_win_prob:
        predicted_winner = player_a_name
        winner_prob = player_a_win_prob
    else:
        predicted_winner = player_b_name
        winner_prob = player_b_win_prob

    return {
        "player_a": player_a_name,
        "player_b": player_b_name,
        "player_a_win_probability": player_a_win_prob,
        "player_b_win_probability": player_b_win_prob,
        "predicted_winner": predicted_winner,
        "winner_probability": winner_prob,
    }


if __name__ == "__main__":
    result = predict_match(
        player_a_name="Carlos Alcaraz",
        player_b_name="Jannik Sinner",
        player_a_rank=2,
        player_b_rank=1,
        player_a_rank_points=7720,
        player_b_rank_points=9930,
        player_a_age=23,
        player_b_age=24,
        player_a_overall_elo=1800,
        player_b_overall_elo=1850,
        player_a_surface_elo=1850,
        player_b_surface_elo=1780,
        surface="Clay",
        best_of=5,
        player_a_recent_win_rate=0.8,
        player_b_recent_win_rate=0.8,
    )

    print(result)