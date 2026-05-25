import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from data_loader import load_matches
from features import create_features, FEATURE_COLUMNS


def build_retrieval_dataset():
    matches = load_matches()
    matches = matches.sort_values("tourney_date").reset_index(drop=True)

    features = create_features(matches)

    # create_features creates winner rows first, loser rows second
    features = features.iloc[: len(matches)].reset_index(drop=True)

    retrieval_df = matches[
        [
            "winner_name",
            "loser_name",
            "surface",
            "tourney_name",
            "tourney_date",
            "score",
        ]
    ].copy()

    for col in FEATURE_COLUMNS:
        retrieval_df[col] = features[col]

    retrieval_df = retrieval_df.dropna(subset=FEATURE_COLUMNS)

    return retrieval_df


def find_similar_matches(query_features: pd.DataFrame, top_k: int = 5):
    retrieval_df = build_retrieval_dataset()

    X = retrieval_df[FEATURE_COLUMNS]
    q = query_features[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    q_scaled = scaler.transform(q)

    similarities = cosine_similarity(q_scaled, X_scaled)[0]

    retrieval_df = retrieval_df.copy()
    retrieval_df["similarity"] = similarities

    return retrieval_df.sort_values("similarity", ascending=False).head(top_k)


if __name__ == "__main__":
    query = pd.DataFrame(
        [
            {
                "rank_diff": -1,
                "rank_points_diff": 1500,
                "age_diff": -5,
                "best_of": 5,
                "is_clay": 1,
                "is_hard": 0,
                "is_grass": 0,
                "recent_win_rate_diff": 0.1,
            }
        ]
    )

    print(find_similar_matches(query))