from datetime import timedelta

import pandas as pd

from data_loader import load_matches
from elo import build_current_elo_table
from draw_loader import load_roland_garros_draw


OUTPUT_PATH = "data/player_stats.csv"
CURRENT_RANKINGS_PATH = "data/current_rankings.csv"


def compute_recent_win_rate(results, window=10):
    recent = results[-window:]
    return sum(recent) / len(recent) if recent else 0.5


def get_latest_historical_info(matches):
    latest = {}

    matches = matches.sort_values("tourney_date")

    for _, row in matches.iterrows():
        latest[row["winner_name"]] = {
            "rank": row.get("winner_rank", 999),
            "rank_points": row.get("winner_rank_points", 0),
            "age": row.get("winner_age", 25),
        }

        latest[row["loser_name"]] = {
            "rank": row.get("loser_rank", 999),
            "rank_points": row.get("loser_rank_points", 0),
            "age": row.get("loser_age", 25),
        }

    return latest


def build_player_stats():
    matches = load_matches()
    matches = matches.sort_values("tourney_date").reset_index(drop=True)

    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"],
        format="%Y%m%d",
    )

    latest_date = matches["tourney_date"].max()
    fatigue_window_days = 14
    fatigue_cutoff = latest_date - timedelta(days=fatigue_window_days)

    current_rankings = pd.read_csv(CURRENT_RANKINGS_PATH)
    elo_table = build_current_elo_table(matches)
    latest_historical_info = get_latest_historical_info(matches)

    try:
        draw = load_roland_garros_draw()
        draw_players = set(draw["player_a"].dropna()) | set(draw["player_b"].dropna())
    except Exception:
        draw_players = set()

    ranking_players = set(current_rankings["player_name"].dropna())
    all_players = sorted(ranking_players | draw_players)

    players_in_matches = pd.unique(
        pd.concat([matches["winner_name"], matches["loser_name"]], ignore_index=True)
    )

    history = {p: [] for p in players_in_matches}
    clay_history = {p: [] for p in players_in_matches}
    serve_points_won = {p: [] for p in players_in_matches}
    return_points_won = {p: [] for p in players_in_matches}

    for _, row in matches.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        surface = row["surface"]

        history[winner].append(1)
        history[loser].append(0)

        if surface == "Clay":
            clay_history[winner].append(1)
            clay_history[loser].append(0)

        if (
            "w_svpt" in matches.columns
            and pd.notna(row.get("w_svpt"))
            and pd.notna(row.get("l_svpt"))
            and row.get("w_svpt", 0) > 0
            and row.get("l_svpt", 0) > 0
        ):
            w_svpt = row["w_svpt"]
            l_svpt = row["l_svpt"]

            w_serve_won = row["w_1stWon"] + row["w_2ndWon"]
            l_serve_won = row["l_1stWon"] + row["l_2ndWon"]

            winner_serve = w_serve_won / w_svpt
            loser_serve = l_serve_won / l_svpt

            serve_points_won[winner].append(winner_serve)
            serve_points_won[loser].append(loser_serve)

            return_points_won[winner].append(1 - loser_serve)
            return_points_won[loser].append(1 - winner_serve)

    rows = []

    for player in all_players:
        ranking_row = current_rankings[current_rankings["player_name"] == player]

        if not ranking_row.empty:
            rank = ranking_row.iloc[0]["rank"]
            rank_points = ranking_row.iloc[0]["rank_points"]
            age = ranking_row.iloc[0]["age"]
            data_source = "current_rankings"
        else:
            hist = latest_historical_info.get(player, {})
            rank = hist.get("rank", 999)
            rank_points = hist.get("rank_points", 0)
            age = hist.get("age", 25)
            data_source = "historical_fallback"

        elo_row = elo_table[elo_table["player_name"] == player]

        if not elo_row.empty:
            overall_elo = float(elo_row.iloc[0]["overall_elo"])
            clay_elo = float(elo_row.iloc[0]["clay_elo"])
            hard_elo = float(elo_row.iloc[0]["hard_elo"])
            grass_elo = float(elo_row.iloc[0]["grass_elo"])
        else:
            overall_elo = clay_elo = hard_elo = grass_elo = 1500.0

        recent_serve = serve_points_won.get(player, [])[-20:]
        recent_return = return_points_won.get(player, [])[-20:]

        serve_strength = sum(recent_serve) / len(recent_serve) if recent_serve else 0.5
        return_strength = (
            sum(recent_return) / len(recent_return) if recent_return else 0.5
        )

        recent_player_matches = matches[
            (
                (matches["winner_name"] == player)
                | (matches["loser_name"] == player)
            )
            & (matches["tourney_date"] >= fatigue_cutoff)
        ]

        rows.append(
            {
                "player_name": player,
                "rank": rank,
                "rank_points": rank_points,
                "age": age,
                "recent_win_rate": compute_recent_win_rate(history.get(player, [])),
                "recent_clay_win_rate": compute_recent_win_rate(
                    clay_history.get(player, [])
                ),
                "overall_elo": overall_elo,
                "clay_elo": clay_elo,
                "hard_elo": hard_elo,
                "grass_elo": grass_elo,
                "serve_strength": serve_strength,
                "return_strength": return_strength,
                "recent_match_count": len(recent_player_matches),
                "fatigue_window_days": fatigue_window_days,
                "matches_played": len(history.get(player, [])),
                "data_source": data_source,
            }
        )

    player_stats = pd.DataFrame(rows).sort_values("rank")
    player_stats.to_csv(OUTPUT_PATH, index=False)

    print(player_stats.head(30))
    print(f"\nSaved player stats to {OUTPUT_PATH}")

    fallback_count = (player_stats["data_source"] == "historical_fallback").sum()
    print(f"Historical fallback players: {fallback_count}")
    print(f"Fatigue window: last {fatigue_window_days} days before {latest_date.date()}")

    return player_stats


if __name__ == "__main__":
    build_player_stats()