import pandas as pd

from data_loader import load_matches


INITIAL_ELO = 1500.0
K_FACTOR = 32.0


def expected_score(player_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400.0))


def update_elo(winner_elo: float, loser_elo: float, k: float = K_FACTOR):
    winner_expected = expected_score(winner_elo, loser_elo)
    loser_expected = expected_score(loser_elo, winner_elo)

    new_winner_elo = winner_elo + k * (1.0 - winner_expected)
    new_loser_elo = loser_elo + k * (0.0 - loser_expected)

    return new_winner_elo, new_loser_elo


def add_elo_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches = matches.sort_values("tourney_date").reset_index(drop=True)

    overall_elo = {}
    surface_elo = {}

    winner_overall_elos = []
    loser_overall_elos = []
    winner_surface_elos = []
    loser_surface_elos = []

    for _, row in matches.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        surface = row["surface"]

        winner_overall = overall_elo.get(winner, INITIAL_ELO)
        loser_overall = overall_elo.get(loser, INITIAL_ELO)

        winner_surface = surface_elo.get((winner, surface), INITIAL_ELO)
        loser_surface = surface_elo.get((loser, surface), INITIAL_ELO)

        winner_overall_elos.append(winner_overall)
        loser_overall_elos.append(loser_overall)
        winner_surface_elos.append(winner_surface)
        loser_surface_elos.append(loser_surface)

        new_winner_overall, new_loser_overall = update_elo(
            winner_overall,
            loser_overall,
        )

        new_winner_surface, new_loser_surface = update_elo(
            winner_surface,
            loser_surface,
        )

        overall_elo[winner] = new_winner_overall
        overall_elo[loser] = new_loser_overall

        surface_elo[(winner, surface)] = new_winner_surface
        surface_elo[(loser, surface)] = new_loser_surface

    matches["winner_overall_elo"] = winner_overall_elos
    matches["loser_overall_elo"] = loser_overall_elos
    matches["winner_surface_elo"] = winner_surface_elos
    matches["loser_surface_elo"] = loser_surface_elos

    return matches


def build_current_elo_table(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    if matches is None:
        matches = load_matches()

    matches = matches.copy()
    matches = matches.sort_values("tourney_date").reset_index(drop=True)

    overall_elo = {}
    surface_elo = {}

    for _, row in matches.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        surface = row["surface"]

        winner_overall = overall_elo.get(winner, INITIAL_ELO)
        loser_overall = overall_elo.get(loser, INITIAL_ELO)

        winner_surface = surface_elo.get((winner, surface), INITIAL_ELO)
        loser_surface = surface_elo.get((loser, surface), INITIAL_ELO)

        new_winner_overall, new_loser_overall = update_elo(
            winner_overall,
            loser_overall,
        )

        new_winner_surface, new_loser_surface = update_elo(
            winner_surface,
            loser_surface,
        )

        overall_elo[winner] = new_winner_overall
        overall_elo[loser] = new_loser_overall

        surface_elo[(winner, surface)] = new_winner_surface
        surface_elo[(loser, surface)] = new_loser_surface

    rows = []

    players = sorted(overall_elo.keys())

    for player in players:
        rows.append(
            {
                "player_name": player,
                "overall_elo": overall_elo.get(player, INITIAL_ELO),
                "clay_elo": surface_elo.get((player, "Clay"), INITIAL_ELO),
                "hard_elo": surface_elo.get((player, "Hard"), INITIAL_ELO),
                "grass_elo": surface_elo.get((player, "Grass"), INITIAL_ELO),
            }
        )

    elo_table = pd.DataFrame(rows)

    return elo_table


if __name__ == "__main__":
    matches = load_matches()

    matches_with_elo = add_elo_features(matches)
    print(matches_with_elo[
        [
            "tourney_date",
            "winner_name",
            "loser_name",
            "surface",
            "winner_overall_elo",
            "loser_overall_elo",
            "winner_surface_elo",
            "loser_surface_elo",
        ]
    ].tail(20))

    elo_table = build_current_elo_table(matches)
    elo_table = elo_table.sort_values("overall_elo", ascending=False)

    print("\nTop overall Elo:")
    print(elo_table.head(20))

    print("\nTop clay Elo:")
    print(elo_table.sort_values("clay_elo", ascending=False).head(20))