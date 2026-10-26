"""Campus Facility Reservation Monitoring - Streamlit deployment of the Midterm model."""
import joblib
import pandas as pd
import streamlit as st

from modules import dashboard, data_processing as dp, exploratory_analysis, prediction, system_design

st.set_page_config(page_title="SAD Predictive System", page_icon="🏫", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(dp.BASE_DIR / "models" / "trained_model.joblib")


@st.cache_data
def load_meta():
    return dp.load_metadata()


@st.cache_data
def load_default():
    return pd.read_csv(dp.DEFAULT_DATA)


@st.cache_data
def clean_cached(df):
    return dp.clean_dataset(df, dp.load_metadata())


@st.cache_data
def score_cached(df):
    return dashboard.score(load_model(), df)


try:
    model = load_model()
    meta = load_meta()
except Exception as e:  # version mismatch / missing file
    st.error("The saved model could not be loaded. Run `python train_model.py` with the same "
             f"scikit-learn version as requirements.txt, then redeploy.\n\nDetails: {e}")
    st.stop()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Facility Reservation Monitoring")
menu = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction",
                                       "Dashboard", "System Design"])
role = st.sidebar.selectbox("Role (demo)", list(system_design.PERMISSIONS), index=2)
perm = system_design.PERMISSIONS[role]
st.sidebar.caption(f"Model v{meta['model_version']} | sklearn {meta['sklearn_version']}")
if meta["sklearn_version"] != __import__("sklearn").__version__:
    st.sidebar.warning("scikit-learn version differs from the one used for training.")

# Active dataset: uploaded (validated) or bundled synthetic data
active_raw = st.session_state.get("active_df")
source = st.session_state.get("active_source", "Bundled midterm dataset (synthetic)")
if active_raw is None:
    active_raw, source = load_default(), "Bundled midterm dataset (synthetic)"
clean_df, clean_log = clean_cached(active_raw)
scored_df = score_cached(clean_df) if len(clean_df) else None

# ---------------------------------------------------------------- pages
if menu == "Home":
    st.title("🏫 Campus Facility Reservation Monitoring")
    st.write("Deployment of the **Midterm data-mining model** that predicts whether a facility "
             "reservation will be **Completed** or **Not Completed** (cancelled / no-show), so "
             "administrators can follow up early.")
    c = st.columns(4)
    c[0].metric("Reservations loaded", len(clean_df))
    c[1].metric("Model", "Decision Tree d=4")
    c[2].metric("Test accuracy", f"{meta['metrics']['accuracy']:.1%}")
    c[3].metric("Role", role)
    st.markdown(
        "**Modules:** Dataset (upload & validate) -> Data Analysis (EDA) -> Prediction (model) -> "
        "Dashboard (risk queue & human decision) -> System Design (SAD).")
    st.info(f"Active data source: {source}")
    st.warning("Predictions support, and never replace, the Facility Administrator's decision. "
               "Use only synthetic or authorised data on the public deployment.")
    st.caption("Differences from the midterm: encoders and model are saved in one scikit-learn "
               "Pipeline (joblib) instead of separate LabelEncoders; the model is not retrained "
               "on refresh; the same features, split and hyper-parameters are kept.")

elif menu == "Dataset":
    st.header("Dataset Upload and Validation")
    if not perm["upload"]:
        st.error("Your role cannot upload data.")
    else:
        up = st.file_uploader("Upload reservation CSV", type="csv")
        if up is not None:
            df, err = dp.read_csv_safely(up)
            if err:
                st.error(err)
            else:
                res = dp.validate_schema(df, meta)
                for e in res.errors:
                    st.error(e)
                for w in res.warnings:
                    st.warning(w)
                if res.ok:
                    st.session_state["active_df"], st.session_state["active_source"] = df, f"Uploaded: {up.name}"
                    st.success(f"'{up.name}' accepted and set as the active dataset. Continue to Data Analysis.")
                    active_raw, source = df, f"Uploaded: {up.name}"
                    clean_df, clean_log = clean_cached(df)
        if st.session_state.get("active_df") is not None and st.button("Reset to bundled dataset"):
            for k in ("active_df", "active_source"):
                st.session_state.pop(k, None)
            st.rerun()
    st.caption(f"Active source: {source}")
    st.subheader("Preview")
    st.dataframe(active_raw.head(20))
    q = dp.quality_summary(active_raw)
    c = st.columns(4)
    c[0].metric("Rows", q["rows"]); c[1].metric("Columns", q["columns"])
    c[2].metric("Duplicate rows", q["duplicate_rows"]); c[3].metric("Null cells", q["total_nulls"])
    if q["nulls_by_column"]:
        st.write("Nulls by column", q["nulls_by_column"])
    with st.expander("Cleaning applied (midterm rules)"):
        for line in clean_log:
            st.write("- " + line)
    st.write("Required columns:", ", ".join(dp.FEATURE_COLUMNS))

elif menu == "Data Analysis":
    exploratory_analysis.render(st, clean_df if len(clean_df) else None)

elif menu == "Prediction":
    prediction.render(st, model, meta, perm["predict"])

elif menu == "Dashboard":
    dashboard.render(st, scored_df, meta, perm["view"], perm["decide"], role)

else:
    system_design.render(st, meta)
