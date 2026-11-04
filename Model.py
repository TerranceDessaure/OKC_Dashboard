"""Data loading, feature engineering and model training.

Kept free of Streamlit so it can be tested on its own (python model.py).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HOOP_X = 41.75  # hoop sits at (+/-41.75, 0) in the tracking coordinates

RAW_COLS = [
    "outcome", "shottype", "gamestate", "distance", "locationx", "locationy",
    "closestdefdist", "closestdefapproach", "contested", "num_contesters",
    "shotclock", "dribblesbefore", "shooterspeed", "three",
]

FEATURES = [
    "distance", "closestdefdist", "contested", "shotclock", "dribblesbefore",
    "shooterspeed", "closing_speed", "is_catch_and_shoot", "is_rushed",
    "num_contesters",
    "gamestate_DIRECTSECONDCHANCE", "gamestate_TRANSITION",  # HALFCOURT = reference
    "shottype_dunk", "shottype_floater", "shottype_heave", "shottype_layup",
    "shottype_post", "shottype_tip",  # jumper = reference
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same engineered features as the original notebook."""
    approach = (
        df["closestdefapproach"].str.strip("{}").str.split(",", expand=True).astype(float)
    )
    return df.assign(
        # positive = defender got closer over the last 4 frames
        closing_speed=approach[0] - approach[3],
        is_catch_and_shoot=df["dribblesbefore"].eq(0),
        is_rushed=df["shotclock"].lt(4),
        # half-court view: hoop at (0, 5.25), baseline at y=0
        court_x=df["locationy"],
        court_y=47 - df["locationx"].abs(),
    )


def load_data(path: str) -> pd.DataFrame:
    return add_features(pd.read_csv(path, usecols=RAW_COLS))


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode, then force the exact training column order.

    reindex() matters for the simulator: a single shot only has one shottype,
    so get_dummies alone would produce missing columns.
    """
    X = pd.get_dummies(df, columns=["shottype", "gamestate"])
    return X.reindex(columns=FEATURES, fill_value=0).astype(float)


def train(df: pd.DataFrame) -> dict:
    X, y = design_matrix(df), df["outcome"].astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Pipeline = scaler is fit on training rows only (no leakage)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    pipe.fit(X_tr, y_tr)

    p = pipe.predict_proba(X_val)[:, 1]
    base = np.full(len(y_val), y_tr.mean())
    metrics = {
        "auc": roc_auc_score(y_val, p),
        "accuracy": accuracy_score(y_val, p > 0.5),
        "log_loss": log_loss(y_val, p),
        "brier": brier_score_loss(y_val, p),
        "base_accuracy": max(y_val.mean(), 1 - y_val.mean()),
        "base_log_loss": log_loss(y_val, base),
        "n_train": len(y_tr),
        "n_val": len(y_val),
    }

    coefs = pd.DataFrame(
        {"feature": FEATURES, "coefficient": pipe[-1].coef_[0]}
    ).sort_values("coefficient", key=abs, ascending=False)

    # Calibration: in each predicted-probability decile, how often did shots go in?
    cal = pd.DataFrame({"pred": p, "actual": y_val.values})
    cal["bin"] = pd.qcut(cal["pred"], 10, duplicates="drop")
    calibration = cal.groupby("bin", observed=True)[["pred", "actual"]].mean()

    return {"model": pipe, "metrics": metrics, "coefs": coefs, "calibration": calibration}


def predict_one(model, shot: dict) -> float:
    row = add_features(pd.DataFrame([shot]))
    return float(model.predict_proba(design_matrix(row))[0, 1])


if __name__ == "__main__":  # quick self-check: python model.py
    df = load_data("training.csv.gz")
    out = train(df)
    m = out["metrics"]
    print({k: round(v, 4) for k, v in m.items()})
    assert m["auc"] > 0.6 and m["accuracy"] > m["base_accuracy"]

    base = dict(distance=3, closestdefdist=6, closestdefapproach="{8,7.5,7,6}",
                contested=False, num_contesters=0, shotclock=14, dribblesbefore=1,
                shooterspeed=4, shottype="dunk", gamestate="HALFCOURT",
                locationx=-40, locationy=0, three=False)
    dunk = predict_one(out["model"], base)
    heave = predict_one(out["model"], {**base, "distance": 40, "shottype": "heave"})
    print(f"open dunk {dunk:.2f}  heave {heave:.2f}")
    assert dunk > 0.7 > 0.2 > heave
    print("ok")