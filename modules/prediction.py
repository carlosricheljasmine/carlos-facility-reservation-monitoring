"""Model inference helpers (pure) + the Streamlit prediction page renderer."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from modules.data_processing import (BASE_DIR, CATEGORICAL, FEATURE_COLUMNS,
                                     NUMERIC)

POSITIVE = "Not Completed"


# ---------------------------------------------------------------- pure logic
def build_input_frame(values: dict) -> pd.DataFrame:
    """One-row DataFrame with the exact training feature names, order and dtypes."""
    row = {c: [str(values[c])] for c in CATEGORICAL}
    row.update({c: [float(values[c])] for c in NUMERIC})
    return pd.DataFrame(row)[FEATURE_COLUMNS]


def validate_inputs(values: dict, meta: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block prediction."""
    errors, warns = [], []
    for c in CATEGORICAL:
        if values.get(c) not in meta["allowed_categories"][c]:
            errors.append(f"{c}: '{values.get(c)}' is not a known category.")
    lt, pp = values.get("Lead_Time_Days"), values.get("Participants")
    try:
        lt, pp = float(lt), float(pp)
    except (TypeError, ValueError):
        return errors + ["Lead time and participants must be numbers."], warns
    if lt != lt or pp != pp:
        return errors + ["Lead time and participants are required."], warns
    if lt < 0:
        errors.append("Lead_Time_Days must be 0 or greater (midterm validation rule).")
    if pp <= 0:
        errors.append("Participants must be greater than 0 (midterm validation rule).")
    for name, v in (("Lead_Time_Days", lt), ("Participants", pp)):
        lo, hi = meta["training_ranges"][name]
        if not errors and (v < lo or v > hi):
            warns.append(f"{name}={v:g} is outside the training range [{lo:g}, {hi:g}]; "
                         "the prediction is an extrapolation and less reliable.")
    return errors, warns


def predict_frame(pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Predicted label + P(Not Completed) for every row."""
    X = frame[FEATURE_COLUMNS].copy()
    X[NUMERIC] = X[NUMERIC].astype(float)
    out = pd.DataFrame(index=frame.index)
    out["Predicted_Outcome"] = pipeline.predict(X)
    if hasattr(pipeline, "predict_proba"):
        idx = list(pipeline.classes_).index(POSITIVE)
        out["Risk_Probability"] = pipeline.predict_proba(X)[:, idx]
    return out


def risk_level(p: float) -> str:
    """Thresholds chosen from the distribution of the saved tree's leaf probabilities."""
    if p >= 0.50:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


def midterm_consistency(pipeline) -> tuple[pd.DataFrame, bool]:
    cases = pd.read_csv(BASE_DIR / "models" / "midterm_test_cases.csv")
    pred = predict_frame(pipeline, cases)["Predicted_Outcome"].values
    cases["app_prediction"] = pred
    cases["match"] = cases["app_prediction"] == cases["midterm_prediction"]
    return cases, bool(cases["match"].all())


# ------------------------------------------------------------ Streamlit page
def render(st, pipeline, meta: dict, role_can_predict: bool):
    import plotly.express as px

    st.header("Predictive Modeling")
    st.caption(f"Model: **{meta['model_name']}** | version **{meta['model_version']}** | "
               f"trained {meta['trained_at_utc']} | scikit-learn {meta['sklearn_version']}")
    st.warning("Predictions are **decision-support outputs only**. They never approve, reject or "
               "cancel a reservation. A Facility Administrator makes the final decision.")
    if not role_can_predict:
        st.error("Your current role cannot run predictions. Switch role in the sidebar (demo).")
        return

    tab1, tab2, tab3 = st.tabs(["Single prediction", "Model metrics", "Consistency & limitations"])

    with tab1:
        allowed = meta["allowed_categories"]
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            user = c1.selectbox("User category", allowed["User_Category"])
            fac = c2.selectbox("Facility type", allowed["Facility_Type"])
            slot = c3.selectbox("Time slot", allowed["Time_Slot"])
            day = c1.selectbox("Day of week", allowed["Day_of_Week"])
            lead = c2.number_input("Lead time (days before use)", value=3.0, step=1.0,
                                   help="Known at booking time. Must be >= 0.")
            part = c3.number_input("Expected participants", value=25.0, step=5.0,
                                   help="Declared at booking time. Must be > 0.")
            submitted = st.form_submit_button("Predict outcome")
        if submitted:
            values = {"User_Category": user, "Facility_Type": fac, "Time_Slot": slot,
                      "Day_of_Week": day, "Lead_Time_Days": lead, "Participants": part}
            errors, warns = validate_inputs(values, meta)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                for w in warns:
                    st.warning(w)
                frame = build_input_frame(values)
                res = predict_frame(pipeline, frame).iloc[0]
                prob = float(res["Risk_Probability"])
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted outcome", res["Predicted_Outcome"])
                m2.metric("P(Not Completed)", f"{prob:.1%}")
                m3.metric("Risk level", risk_level(prob))
                st.caption(f"Model v{meta['model_version']} | prediction time {ts} | "
                           "probability = share of training rows in the tree leaf (a rough estimate, "
                           "not a calibrated probability).")
                st.info("Recommended human follow-up: " + follow_up_text(risk_level(prob)))
                log = st.session_state.setdefault("prediction_log", [])
                log.append({"timestamp": ts, "model_version": meta["model_version"], **values,
                            "predicted": res["Predicted_Outcome"], "p_not_completed": round(prob, 3),
                            "risk_level": risk_level(prob)})
                with st.expander("Exact input row sent to the pipeline"):
                    st.dataframe(frame)
        log = st.session_state.get("prediction_log", [])
        if log:
            st.subheader("Predictions made this session")
            ldf = pd.DataFrame(log)
            st.dataframe(ldf)
            st.download_button("Download prediction log (CSV)", ldf.to_csv(index=False),
                               "prediction_log.csv", "text/csv")

    with tab2:
        m = meta["metrics"]
        c = st.columns(5)
        c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        c[1].metric("Precision (wtd)", f"{m['precision_weighted']:.3f}")
        c[2].metric("Recall (wtd)", f"{m['recall_weighted']:.3f}")
        c[3].metric("F1 (wtd)", f"{m['f1_weighted']:.3f}")
        c[4].metric("Majority baseline", f"{meta['majority_baseline_accuracy']:.3f}")
        st.caption(f"Hold-out test set: {meta['n_test']} of {meta['n_rows']} rows (20%, stratified, "
                   f"random_state={meta['random_state']}) - identical to the midterm.")
        labels = meta["classes"]
        cm = pd.DataFrame(meta["confusion_matrix"], index=[f"Actual {l}" for l in labels],
                          columns=[f"Pred {l}" for l in labels])
        a, b = st.columns(2)
        a.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                 title="Confusion matrix (test set)"))
        imp = (pd.Series(meta["feature_importances"]).sort_values()
               .rename_axis("feature").reset_index(name="importance"))
        b.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                              title="Feature importance"))
        st.dataframe(pd.DataFrame(meta["per_class"]).T.round(3))

    with tab3:
        st.subheader("Midterm consistency check")
        st.write("Runs the saved pipeline on the 80 midterm test rows and compares each prediction "
                 "with the label produced by the original midterm LabelEncoder + DecisionTree code.")
        if st.button("Run consistency check"):
            cases, ok = midterm_consistency(pipeline)
            (st.success if ok else st.error)(
                f"{int(cases['match'].sum())}/{len(cases)} predictions match the midterm model.")
            st.dataframe(cases.head(20))
        st.subheader("Limitations")
        base = meta["majority_baseline_accuracy"]
        st.markdown(
            f"- Test accuracy ({m['accuracy']:.1%}) is **not better than always predicting "
            f"'Completed' ({base:.1%})**; the model finds only weak patterns in this data.\n"
            "- The dataset is synthetic with 400 rows and four hand-picked lead-time values per "
            "category, so results should not be generalised to real campus behaviour.\n"
            "- Recall for 'Not Completed' is low, so many at-risk reservations are missed.\n"
            "- Only booking-time fields are used. Usage_Status, Actual_Attendance and "
            "Attendance_Ratio are excluded as leakage.\n"
            "- Do not use a prediction alone to deny or penalise a student, staff or external user.")


def follow_up_text(level: str) -> str:
    return {
        "High": "Send a confirmation reminder 48 h before the slot and ask the requester to reconfirm; "
                "the administrator may hold a short waitlist for this facility.",
        "Medium": "Send a standard reminder one day before the reservation.",
        "Low": "No extra action; routine reminder only.",
    }[level]
