import pandas as pd

from draw_loader import load_roland_garros_draw


PLAYER_STATS_PATH = "data/player_stats.csv"


def main():
    draw = load_roland_garros_draw()
    player_stats = pd.read_csv(PLAYER_STATS_PATH)

    draw_players = sorted(
        set(draw["player_a"].dropna()) | set(draw["player_b"].dropna())
    )

    known_players = set(player_stats["player_name"].dropna())

    missing = [p for p in draw_players if p not in known_players]
    matched = [p for p in draw_players if p in known_players]

    print(f"Players in draw: {len(draw_players)}")
    print(f"Matched players: {len(matched)}")
    print(f"Missing players: {len(missing)}")

    print("\nMissing players:")
    for p in missing:
        print("-", p)

    pd.DataFrame({"missing_player": missing}).to_csv(
        "data/missing_draw_players.csv",
        index=False,
    )

    print("\nSaved missing players to data/missing_draw_players.csv")


if __name__ == "__main__":
    main()