import pandas as pd

from elo import add_elo_features


FEATURE_COLUMNS = [
    "rank_diff",
    "rank_points_diff",
    "age_diff",
    "best_of",
    "is_clay",
    "is_hard",
    "is_grass",
    "recent_win_rate_diff",
    "overall_elo_diff",
    "surface_elo_diff",
]


def compute_recent_win_rates(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("tourney_date").reset_index(drop=True)

    players = pd.unique(
        pd.concat([df["winner_name"], df["loser_name"]], ignore_index=True)
    )

    history = {player: [] for player in players}

    winner_recent_rates = []
    loser_recent_rates = []

    for _, row in df.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]

        winner_history = history[winner][-window:]
        loser_history = history[loser][-window:]

        winner_recent_rate = (
            sum(winner_history) / len(winner_history)
            if len(winner_history) > 0
            else 0.5
        )

        loser_recent_rate = (
            sum(loser_history) / len(loser_history)
            if len(loser_history) > 0
            else 0.5
        )

        winner_recent_rates.append(winner_recent_rate)
        loser_recent_rates.append(loser_recent_rate)

        history[winner].append(1)
        history[loser].append(0)

    df["winner_recent_win_rate"] = winner_recent_rates
    df["loser_recent_win_rate"] = loser_recent_rates

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("tourney_date").reset_index(drop=True)

    df = df.dropna(
        subset=[
            "winner_name",
            "loser_name",
            "winner_rank",
            "loser_rank",
            "winner_rank_points",
            "loser_rank_points",
            "winner_age",
            "loser_age",
            "surface",
            "best_of",
        ]
    )

    df = compute_recent_win_rates(df)
    df = add_elo_features(df)

    is_clay = (df["surface"] == "Clay").astype(int)
    is_hard = (df["surface"] == "Hard").astype(int)
    is_grass = (df["surface"] == "Grass").astype(int)

    winners = pd.DataFrame(
        {
            "rank_diff": df["winner_rank"] - df["loser_rank"],
            "rank_points_diff": df["winner_rank_points"] - df["loser_rank_points"],
            "age_diff": df["winner_age"] - df["loser_age"],
            "best_of": df["best_of"],
            "is_clay": is_clay,
            "is_hard": is_hard,
            "is_grass": is_grass,
            "recent_win_rate_diff": (
                df["winner_recent_win_rate"] - df["loser_recent_win_rate"]
            ),
            "overall_elo_diff": df["winner_overall_elo"] - df["loser_overall_elo"],
            "surface_elo_diff": df["winner_surface_elo"] - df["loser_surface_elo"],
            "target": 1,
        }
    )

    losers = pd.DataFrame(
        {
            "rank_diff": df["loser_rank"] - df["winner_rank"],
            "rank_points_diff": df["loser_rank_points"] - df["winner_rank_points"],
            "age_diff": df["loser_age"] - df["winner_age"],
            "best_of": df["best_of"],
            "is_clay": is_clay,
            "is_hard": is_hard,
            "is_grass": is_grass,
            "recent_win_rate_diff": (
                df["loser_recent_win_rate"] - df["winner_recent_win_rate"]
            ),
            "overall_elo_diff": df["loser_overall_elo"] - df["winner_overall_elo"],
            "surface_elo_diff": df["loser_surface_elo"] - df["winner_surface_elo"],
            "target": 0,
        }
    )

    dataset = pd.concat([winners, losers], ignore_index=True)

    return dataset


if __name__ == "__main__":
    from data_loader import load_matches

    matches = load_matches()
    dataset = create_features(matches)

    print(dataset.head())
    print(dataset.shape)
    print(dataset[FEATURE_COLUMNS].describe())
    print(dataset["target"].value_counts(normalize=True))