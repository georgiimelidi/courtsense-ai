from pathlib import Path

import pandas as pd
import requests


OUTPUT_PATH = Path("data/current_rankings.csv")
ESPN_RANKINGS_URL = "https://www.espn.com/tennis/rankings"


def fetch_current_rankings() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        ESPN_RANKINGS_URL,
        headers=headers,
        timeout=20,
    )
    response.raise_for_status()

    tables = pd.read_html(response.text)

    print(f"Found {len(tables)} tables")

    for i, table in enumerate(tables):
        print(f"\nTable {i}")
        print(table.head())
        print(table.columns)

    # Usually the ranking table is the largest table
    df = max(tables, key=len)

    print("\nSelected table:")
    print(df.head())
    print(df.columns)

    # Normalize columns
    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "rk": "rank",
        "rank": "rank",
        "player": "player_name",
        "name": "player_name",
        "points": "rank_points",
        "pts": "rank_points",
        "age": "age",
    }

    df = df.rename(columns=rename_map)

    needed = ["rank", "player_name", "rank_points", "age"]

    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}. Available columns: {df.columns.tolist()}"
        )

    df = df[needed].copy()

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["rank_points"] = (
        df["rank_points"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df = df.dropna(subset=["rank", "player_name", "rank_points", "age"])

    df["rank"] = df["rank"].astype(int)
    df["rank_points"] = df["rank_points"].astype(int)
    df["age"] = df["age"].astype(int)

    df = df.sort_values("rank").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal rankings:")
    print(df.head(30))
    print(f"\nSaved current rankings to {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    fetch_current_rankings()