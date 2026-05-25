import pandas as pd

from data_loader import load_matches


def get_head_to_head(player_a: str, player_b: str) -> dict:
    matches = load_matches()

    h2h = matches[
        (
            (matches["winner_name"] == player_a)
            & (matches["loser_name"] == player_b)
        )
        |
        (
            (matches["winner_name"] == player_b)
            & (matches["loser_name"] == player_a)
        )
    ].copy()

    if h2h.empty:
        return {
            "player_a_wins": 0,
            "player_b_wins": 0,
            "clay_player_a_wins": 0,
            "clay_player_b_wins": 0,
            "matches": h2h,
        }

    player_a_wins = (h2h["winner_name"] == player_a).sum()
    player_b_wins = (h2h["winner_name"] == player_b).sum()

    clay_h2h = h2h[h2h["surface"] == "Clay"]

    clay_player_a_wins = (clay_h2h["winner_name"] == player_a).sum()
    clay_player_b_wins = (clay_h2h["winner_name"] == player_b).sum()

    h2h = h2h.sort_values("tourney_date", ascending=False)

    return {
        "player_a_wins": int(player_a_wins),
        "player_b_wins": int(player_b_wins),
        "clay_player_a_wins": int(clay_player_a_wins),
        "clay_player_b_wins": int(clay_player_b_wins),
        "matches": h2h,
    }


def h2h_agent(player_a_name: str, player_b_name: str, surface: str) -> str:
    h2h = get_head_to_head(player_a_name, player_b_name)

    a_wins = h2h["player_a_wins"]
    b_wins = h2h["player_b_wins"]

    if a_wins + b_wins == 0:
        return "Head-to-Head Agent is neutral: no historical meetings found in the dataset."

    if surface == "Clay":
        clay_a = h2h["clay_player_a_wins"]
        clay_b = h2h["clay_player_b_wins"]

        if clay_a + clay_b > 0:
            if clay_a > clay_b:
                return (
                    f"Head-to-Head Agent favors {player_a_name}: "
                    f"overall H2H {a_wins}-{b_wins}, clay H2H {clay_a}-{clay_b}."
                )
            elif clay_b > clay_a:
                return (
                    f"Head-to-Head Agent favors {player_b_name}: "
                    f"overall H2H {a_wins}-{b_wins}, clay H2H {clay_a}-{clay_b}."
                )

    if a_wins > b_wins:
        return f"Head-to-Head Agent favors {player_a_name}: overall H2H {a_wins}-{b_wins}."
    elif b_wins > a_wins:
        return f"Head-to-Head Agent favors {player_b_name}: overall H2H {a_wins}-{b_wins}."

    return f"Head-to-Head Agent is neutral: overall H2H is tied {a_wins}-{b_wins}."