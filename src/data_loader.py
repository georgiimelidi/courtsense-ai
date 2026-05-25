import pandas as pd


FILES = [
    "data/atp_matches_2020.csv",
    "data/atp_matches_2021.csv",
    "data/atp_matches_2022.csv",
    "data/atp_matches_2023.csv",
    "data/atp_matches_2024.csv",
    "data/atp_matches_2025.csv",
    "data/atp_matches_2026.csv",
]

def load_matches():
    dfs = []

    for file in FILES:
        df = pd.read_csv(file)
        dfs.append(df)

    matches = pd.concat(dfs, ignore_index=True)

    return matches


if __name__ == "__main__":
    df = load_matches()

    print(df.shape)
    print(df.columns.tolist())
    print(df.head())
