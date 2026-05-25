import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from xgboost import XGBClassifier

from data_loader import load_matches
from features import create_features, FEATURE_COLUMNS


MODEL_PATH = "model.pkl"


def train_model():
    matches = load_matches()
    dataset = create_features(matches)

    X = dataset[FEATURE_COLUMNS]
    y = dataset["target"]

    print("Feature summary:")
    print(X.describe())

    print("\nTarget distribution:")
    print(y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    pred_labels = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, pred_labels)
    auc = roc_auc_score(y_test, pred_probs)
    loss = log_loss(y_test, pred_probs)

    print("\nModel performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print(f"Log loss: {loss:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_model()