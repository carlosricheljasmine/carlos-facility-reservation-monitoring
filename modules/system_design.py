"""SAD integration page: requirements, actors, use case diagram, workflow."""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent

FUNCTIONAL = [
    ("FR1", "Upload and validate reservation CSV files (required columns, categories, nulls, duplicates).",
     "Midterm Part II found invalid lead times, duplicates and missing values that had to be cleaned.",
     "Dataset"),
    ("FR2", "Report non-completion by lead-time category, user category, facility, time slot and day.",
     "Overall non-completion is 42.75% (171/400); Short Lead bookings reach 49.7% vs 29.4% for Early Booking.",
     "Data Analysis"),
    ("FR3", "Predict P(Not Completed) for a new reservation from booking-time fields only.",
     "A Decision Tree (depth 4) on 6 booking-time features was the midterm model; leakage fields excluded.",
     "Prediction"),
    ("FR4", "Show a ranked risk/review queue with recommended follow-up per risk level.",
     "Cancellations (28.0%) and no-shows (14.75%) both lead to lost facility time.",
     "Dashboard"),
    ("FR5", "Require a named Facility Administrator to record the final decision with note and timestamp; "
            "show model metrics, version and limitations beside every score.",
     "Test accuracy 55.0% is below the 57.5% majority baseline, so scores cannot be trusted as verdicts.",
     "Dashboard / Prediction"),
]
NON_FUNCTIONAL = [
    ("NFR1 Performance", "The model is loaded once with @st.cache_resource and never retrained at runtime; "
                         "scoring 400 rows and rendering a page should take under 3 seconds."),
    ("NFR2 Privacy & security", "Only synthetic / authorised, non-identifying data is used; no credentials or "
                                "student records in the repository; features are role-gated (demo roles)."),
    ("NFR3 Reliability & transparency", "Invalid files and inputs produce clear messages instead of tracebacks; "
                                        "model version, prediction time and limitations are always visible."),
]
ROLES = pd.DataFrame([
    {"Actor": "Requester", "Upload data": "No", "Run prediction": "No", "View risk dashboard": "No", "Record final decision": "No"},
    {"Actor": "Reservation Officer", "Upload data": "Yes", "Run prediction": "Yes", "View risk dashboard": "Yes", "Record final decision": "No"},
    {"Actor": "Facility Administrator", "Upload data": "Yes", "Run prediction": "Yes", "View risk dashboard": "Yes", "Record final decision": "Yes"},
    {"Actor": "Data Analyst / System Admin", "Upload data": "Yes", "Run prediction": "Yes", "View risk dashboard": "Yes", "Record final decision": "No"},
])
PERMISSIONS = {
    "Requester": dict(upload=False, predict=False, view=False, decide=False),
    "Reservation Officer": dict(upload=True, predict=True, view=True, decide=False),
    "Facility Administrator": dict(upload=True, predict=True, view=True, decide=True),
    "Data Analyst / System Admin": dict(upload=True, predict=True, view=True, decide=False),
}


def render(st, meta):
    st.header("System Design (SAD Integration)")
    st.subheader("Functional requirements (evidence-based)")
    st.dataframe(pd.DataFrame(FUNCTIONAL, columns=["ID", "Requirement", "Evidence from midterm data/model",
                                                   "Implemented in page"]))
    st.subheader("Non-functional requirements")
    st.dataframe(pd.DataFrame(NON_FUNCTIONAL, columns=["ID", "Requirement"]))
    st.subheader("Actors and permissions")
    st.dataframe(ROLES)
    st.caption("Roles are simulated with a sidebar selector for the laboratory demo. A production system "
               "would use real authentication (e.g. university SSO).")
    st.subheader("Updated use case diagram")
    st.image(str(BASE / "assets" / "use_case_diagram.png"))
    st.subheader("Analytical process / decision workflow")
    st.image(str(BASE / "assets" / "decision_workflow.png"))
    st.info(f"Prediction checkpoint: at booking time. Inputs: {', '.join(meta['features'])}. "
            f"Excluded as leakage: {', '.join(meta['excluded_leakage_fields'])}.")
