"""Exploratory Data Analysis: statistics, class distribution, interactive Plotly charts."""
from __future__ import annotations

import pandas as pd
import plotly.express as px

COLORS = {"Completed": "#2ca02c", "Not Completed": "#d62728"}
LEAD_ORDER = ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]


def class_distribution(df: pd.DataFrame, col: str = "Reservation_Outcome") -> pd.DataFrame:
    vc = df[col].value_counts().rename_axis(col).reset_index(name="count")
    vc["percent"] = (vc["count"] / vc["count"].sum() * 100).round(1)
    return vc


def non_completion_rate(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by).agg(reservations=("Reservation_Outcome", "size"),
                           non_completed=("Reservation_Outcome", lambda s: (s == "Not Completed").sum()))
    g["non_completion_rate_pct"] = (g["non_completed"] / g["reservations"] * 100).round(1)
    return g.reset_index()


def fig_class_distribution(df, col="Reservation_Outcome"):
    d = class_distribution(df, col)
    return px.bar(d, x=col, y="count", text="percent", color=col,
                  color_discrete_map=COLORS, title=f"Class distribution: {col}")


def fig_volume(df, by):
    d = df[by].value_counts().rename_axis(by).reset_index(name="reservations")
    return px.bar(d, x=by, y="reservations", title=f"Reservation volume by {by}", color=by)


def fig_outcome_by(df, by):
    ct = (pd.crosstab(df[by], df["Reservation_Outcome"], normalize="index") * 100).reset_index()
    long = ct.melt(id_vars=by, var_name="Outcome", value_name="Percent")
    fig = px.bar(long, x=by, y="Percent", color="Outcome", color_discrete_map=COLORS,
                 title=f"Outcome share (%) by {by}", barmode="stack")
    if by == "Lead_Time_Category":
        fig.update_xaxes(categoryorder="array", categoryarray=LEAD_ORDER)
    return fig


def fig_heatmap(df, rows="Time_Slot", cols="Day_of_Week"):
    piv = (df.assign(nc=(df["Reservation_Outcome"] == "Not Completed") * 100)
             .pivot_table(index=rows, columns=cols, values="nc", aggfunc="mean").round(1))
    return px.imshow(piv, text_auto=True, color_continuous_scale="Reds",
                     title=f"Non-completion rate (%): {rows} x {cols}")


def fig_numeric(df, num, color="Reservation_Outcome"):
    return px.histogram(df, x=num, color=color, barmode="overlay", opacity=0.7,
                        color_discrete_map=COLORS, title=f"Distribution of {num} by outcome")


def render(st, df: pd.DataFrame):
    st.header("Exploratory Data Analysis")
    if df is None:
        st.info("Load a dataset on the Dataset page first.")
        return
    has_out = "Reservation_Outcome" in df.columns
    st.subheader("Descriptive statistics")
    a, b = st.columns(2)
    a.write("Numerical variables")
    a.dataframe(df.select_dtypes("number").describe().T.round(2))
    b.write("Categorical variables")
    b.dataframe(df.select_dtypes(exclude="number").describe().T)

    if not has_out:
        st.warning("No outcome column in this dataset, so outcome charts are unavailable.")
        return

    st.subheader("Class distribution")
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_class_distribution(df))
    if "Usage_Status" in df.columns:
        c2.plotly_chart(fig_class_distribution(df, "Usage_Status"))
    dist = class_distribution(df)
    st.caption("Positive-risk class: " + ", ".join(
        f"{r.Reservation_Outcome} = {r['count']} ({r.percent}%)" for _, r in dist.iterrows()))

    st.subheader("Interactive charts")
    cats = ["Facility_Type", "User_Category", "Lead_Time_Category", "Time_Slot", "Day_of_Week"]
    cats = [c for c in cats if c in df.columns]
    t1, t2, t3, t4 = st.tabs(["Outcome by category", "Volume", "Heatmap", "Numeric distribution"])
    with t1:
        by = st.selectbox("Group by", cats, key="eda_by")
        st.plotly_chart(fig_outcome_by(df, by))
        st.dataframe(non_completion_rate(df, by))
    with t2:
        by2 = st.selectbox("Count reservations by", cats, key="eda_vol")
        st.plotly_chart(fig_volume(df, by2))
    with t3:
        r = st.selectbox("Rows", cats, index=cats.index("Time_Slot") if "Time_Slot" in cats else 0, key="hr")
        c = st.selectbox("Columns", cats, index=cats.index("Day_of_Week") if "Day_of_Week" in cats else 1, key="hc")
        if r == c:
            st.warning("Choose two different variables.")
        else:
            st.plotly_chart(fig_heatmap(df, r, c))
    with t4:
        num = st.selectbox("Numeric variable", ["Lead_Time_Days", "Participants"], key="eda_num")
        st.plotly_chart(fig_numeric(df, num))
