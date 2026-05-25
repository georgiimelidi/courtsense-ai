import joblib
import pandas as pd

from features import FEATURE_COLUMNS


MODEL_PATH = "model.pkl"


def get_feature_importance():
    model = joblib.load(MODEL_PATH)

    importances = model.feature_importances_

    df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": importances,
        }
    )

    df = df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    return df