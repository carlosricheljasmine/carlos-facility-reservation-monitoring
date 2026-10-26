"""Re-creates the Midterm model and saves a fitted preprocessing + model pipeline.

Run once (NOT on every Streamlit refresh):
    python train_model.py

Reused from the Midterm notebook (Part IV):
    * feature_cols  = User_Category, Facility_Type, Lead_Time_Days, Participants,
                      Time_Slot, Day_of_Week
    * target        = Reservation_Outcome  (Completed / Not Completed)
    * split         = train_test_split(test_size=0.20, random_state=42, stratify=y)
    * model         = DecisionTreeClassifier(max_depth=4, random_state=42)

Change from the midterm: LabelEncoder objects (fit on the whole dataset, kept in a
dict) are replaced by an OrdinalEncoder inside a scikit-learn Pipeline, so the SAME
preprocessing is applied at training and inference time. Both assign integer codes
in alphabetical order, so the fitted tree is identical; this script asserts it.

Leakage control: Usage_Status, Actual_Attendance, Attendance_Ratio and
Reservation_Outcome are known only AFTER the reservation date and are never features.
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_PATH = "data/cleaned_facility_reservations.csv"
MODEL_PATH = "models/trained_model.joblib"
META_PATH = "models/model_metadata.json"
TEST_CASES_PATH = "models/midterm_test_cases.csv"

CATEGORICAL = ["User_Category", "Facility_Type", "Time_Slot", "Day_of_Week"]
NUMERIC = ["Lead_Time_Days", "Participants"]
FEATURES = ["User_Category", "Facility_Type", "Lead_Time_Days", "Participants",
            "Time_Slot", "Day_of_Week"]
TARGET = "Reservation_Outcome"
LEAKAGE = ["Usage_Status", "Actual_Attendance", "Attendance_Ratio", TARGET]


def main():
    df = pd.read_csv(DATA_PATH)
    X_raw = df[FEATURES].copy()
    X_raw[NUMERIC] = X_raw[NUMERIC].astype(float)
    y = df[TARGET].copy()

    # ---- Midterm method (LabelEncoder) : used only as the reference result ----
    X_mid = X_raw.copy()
    for col in CATEGORICAL:
        X_mid[col] = LabelEncoder().fit_transform(X_mid[col])
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y)
    Xtr_m, Xte_m, ytr_m, yte_m = train_test_split(
        X_mid, y_enc, test_size=0.20, random_state=RANDOM_STATE, stratify=y_enc)
    mid_clf = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE).fit(Xtr_m, ytr_m)
    mid_pred = target_le.inverse_transform(mid_clf.predict(Xte_m))

    # ---- Deployable pipeline (same split indices) ----
    train_idx, test_idx = Xtr_m.index, Xte_m.index
    X_train, X_test = X_raw.loc[train_idx], X_raw.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(), CATEGORICAL)], remainder="passthrough",
        verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    pipe = Pipeline([
        ("preprocess", pre),
        ("model", DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    assert (pred == mid_pred).all(), "Pipeline differs from midterm model!"
    print("Pipeline predictions == midterm predictions on all", len(pred), "test rows")

    labels = list(pipe.classes_)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
    }
    cm = confusion_matrix(y_test, pred, labels=labels)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    imp = dict(zip(names, map(float, pipe.named_steps["model"].feature_importances_)))

    meta = {
        "model_name": "Decision Tree (max_depth=4) - Reservation Outcome",
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "random_state": RANDOM_STATE,
        "dataset": DATA_PATH,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "excluded_leakage_fields": LEAKAGE,
        "target": TARGET,
        "classes": labels,
        "positive_class_for_risk": "Not Completed",
        "allowed_categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL},
        "training_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC},
        "training_medians": {c: float(df[c].median()) for c in NUMERIC},
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "per_class": {k: v for k, v in report.items() if k in labels},
        "feature_importances": imp,
        "majority_baseline_accuracy": float(y_test.value_counts(normalize=True).max()),
    }
    joblib.dump(pipe, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    tc = X_test.copy()
    tc["midterm_prediction"] = mid_pred
    tc["actual_outcome"] = y_test.values
    tc.to_csv(TEST_CASES_PATH, index=False)

    print(json.dumps(metrics, indent=2))
    print("Confusion matrix", labels, "\n", cm)
    print("Majority baseline accuracy:", round(meta["majority_baseline_accuracy"], 4))
    print("Saved:", MODEL_PATH, META_PATH, TEST_CASES_PATH)


if __name__ == "__main__":
    main()
