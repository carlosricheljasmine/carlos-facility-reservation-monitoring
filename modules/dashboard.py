"""Decision-support dashboard: KPIs, facility risk, review queue and human decisions."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px

from modules.prediction import follow_up_text, predict_frame, risk_level

RISK_COLORS = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}
DECISIONS = ["Reminder sent", "Reconfirmation requested", "Waitlist offered",
             "No action needed", "Escalated to department head"]


def score(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    res = predict_frame(pipeline, df)
    out["Predicted_Outcome"] = res["Predicted_Outcome"].values
    out["Risk_Probability"] = res["Risk_Probability"].values.round(3)
    out["Risk_Level"] = out["Risk_Probability"].apply(risk_level)
    if "Reservation_ID" not in out.columns:
        out.insert(0, "Reservation_ID", [f"ROW-{i + 1}" for i in range(len(out))])
    return out


def kpis(scored: pd.DataFrame) -> dict:
    n = len(scored)
    k = {"total": n, "high_risk": int((scored["Risk_Level"] == "High").sum()),
         "avg_risk": float(scored["Risk_Probability"].mean())}
    if "Reservation_Outcome" in scored.columns:
        k["non_completion"] = float((scored["Reservation_Outcome"] == "Not Completed").mean())
    if "Usage_Status" in scored.columns:
        k["no_show"] = float((scored["Usage_Status"] == "No-Show").mean())
        k["cancelled"] = float((scored["Usage_Status"] == "Cancelled").mean())
    if "Attendance_Ratio" in scored.columns and "Usage_Status" in scored.columns:
        used = scored[scored["Usage_Status"] == "Used"]
        k["utilization"] = float(used["Attendance_Ratio"].mean()) if len(used) else 0.0
    return k


def facility_risk(scored: pd.DataFrame) -> pd.DataFrame:
    g = scored.groupby("Facility_Type").agg(
        reservations=("Reservation_ID", "size"),
        avg_predicted_risk=("Risk_Probability", "mean"),
        high_risk=("Risk_Level", lambda s: (s == "High").sum()))
    if "Reservation_Outcome" in scored.columns:
        g["observed_non_completion"] = scored.groupby("Facility_Type")["Reservation_Outcome"].apply(
            lambda s: (s == "Not Completed").mean())
    return g.round(3).reset_index().sort_values("avg_predicted_risk", ascending=False)


def review_queue(scored: pd.DataFrame, levels, facility="All", top_n=25) -> pd.DataFrame:
    q = scored[scored["Risk_Level"].isin(levels)]
    if facility != "All":
        q = q[q["Facility_Type"] == facility]
    cols = ["Reservation_ID", "Facility_Type", "User_Category", "Time_Slot", "Day_of_Week",
            "Lead_Time_Days", "Participants", "Risk_Probability", "Risk_Level"]
    return q.sort_values("Risk_Probability", ascending=False)[cols].head(top_n)


def render(st, scored: pd.DataFrame | None, meta: dict, can_view: bool, can_decide: bool,
           role: str):
    st.header("Decision-Support Dashboard")
    if not can_view:
        st.error("Your current role cannot view risk results.")
        return
    if scored is None:
        st.info("Load a valid dataset on the Dataset page first.")
        return
    st.warning("Scores are advisory. No reservation is rejected or cancelled automatically; the "
               "Facility Administrator records the final follow-up decision.")
    k = kpis(scored)
    cols = st.columns(5)
    cols[0].metric("Reservations scored", k["total"])
    cols[1].metric("High-risk reservations", k["high_risk"])
    cols[2].metric("Avg predicted risk", f"{k['avg_risk']:.1%}")
    if "non_completion" in k:
        cols[3].metric("Observed non-completion", f"{k['non_completion']:.1%}")
    if "utilization" in k:
        cols[4].metric("Utilization (attendance / declared, used slots)", f"{k['utilization']:.1%}")
    if "no_show" in k:
        st.caption(f"Historical cancellations {k['cancelled']:.1%} | no-shows {k['no_show']:.1%}. "
                   "These KPIs describe past outcomes and are not model inputs.")

    a, b = st.columns(2)
    fr = facility_risk(scored)
    a.plotly_chart(px.bar(fr, x="Facility_Type", y="avg_predicted_risk", color="avg_predicted_risk",
                          color_continuous_scale="Reds", title="Average predicted risk by facility"))
    lv = scored["Risk_Level"].value_counts().rename_axis("Risk_Level").reset_index(name="count")
    b.plotly_chart(px.pie(lv, names="Risk_Level", values="count", color="Risk_Level",
                          color_discrete_map=RISK_COLORS, title="Risk-level mix"))
    st.dataframe(fr)

    st.subheader("Interpretation")
    top = fr.iloc[0]
    st.markdown(
        f"- **{top['Facility_Type']}** has the highest average predicted risk "
        f"({top['avg_predicted_risk']:.1%}).\n"
        f"- {k['high_risk']} of {k['total']} reservations are High risk "
        f"(P(Not Completed) >= 50%); Medium is 40-50%.\n"
        "- Differences between facilities are small and the model is only slightly informative "
        "(see Prediction > Model metrics), so treat the queue as a prompt to check, not a verdict.")

    st.subheader("Risk / review queue")
    f1, f2, f3 = st.columns(3)
    levels = f1.multiselect("Risk levels", ["High", "Medium", "Low"], default=["High"])
    fac = f2.selectbox("Facility", ["All"] + sorted(scored["Facility_Type"].unique()))
    top_n = f3.slider("Rows", 5, 100, 25)
    q = review_queue(scored, levels, fac, top_n)
    st.dataframe(q)
    if q.empty:
        st.info("No reservations match the filters.")
        return
    with st.expander("Recommended human follow-up by risk level"):
        for lvl in ["High", "Medium", "Low"]:
            st.write(f"**{lvl}:** {follow_up_text(lvl)}")

    st.subheader("Record decision")
    if not can_decide:
        st.info(f"Role '{role}' can view results but only a Facility Administrator can record "
                "final decisions.")
    else:
        with st.form("decision_form"):
            rid = st.selectbox("Reservation", q["Reservation_ID"].tolist())
            decision = st.selectbox("Decision", DECISIONS)
            reviewer = st.text_input("Reviewer name")
            note = st.text_area("Note / justification")
            if st.form_submit_button("Save decision"):
                if not reviewer.strip():
                    st.error("Reviewer name is required for the audit trail.")
                else:
                    row = q[q["Reservation_ID"] == rid].iloc[0]
                    st.session_state.setdefault("decision_log", []).append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model_version": meta["model_version"], "reservation_id": rid,
                        "risk_probability": row["Risk_Probability"], "risk_level": row["Risk_Level"],
                        "decision": decision, "reviewer": reviewer.strip(), "note": note.strip()})
                    st.success(f"Decision saved for {rid}.")
    log = st.session_state.get("decision_log", [])
    if log:
        st.subheader("Decision log (this session)")
        ldf = pd.DataFrame(log)
        st.dataframe(ldf)
        st.download_button("Download decision log (CSV)", ldf.to_csv(index=False),
                           "decision_log.csv", "text/csv")
