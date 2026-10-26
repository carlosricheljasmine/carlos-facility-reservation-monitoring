"""Run with: python -m pytest tests -q   (no Streamlit needed)."""
import io, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for name in ("plotly", "plotly.express"):          # charts are not under test
    sys.modules.setdefault(name, types.ModuleType(name))

import joblib, pandas as pd, pytest
from modules import data_processing as dp, prediction as pr, dashboard as db

META = dp.load_metadata()
MODEL = joblib.load(dp.BASE_DIR / "models" / "trained_model.joblib")
GOOD = dict(User_Category="Student", Facility_Type="Lab", Time_Slot="Morning",
            Day_of_Week="Weekday", Lead_Time_Days=3, Participants=25)

def test_matches_midterm():
    assert pr.midterm_consistency(MODEL)[1]

def test_valid_input():
    assert pr.validate_inputs(GOOD, META) == ([], [])
    assert pr.predict_frame(MODEL, pr.build_input_frame(GOOD)).shape == (1, 2)

@pytest.mark.parametrize("bad", [{"Lead_Time_Days": -1}, {"Participants": 0},
                                 {"Facility_Type": "Pool"}, {"Lead_Time_Days": float("nan")}])
def test_invalid_input(bad):
    assert pr.validate_inputs({**GOOD, **bad}, META)[0]

def test_extrapolation_warning():
    assert pr.validate_inputs({**GOOD, "Lead_Time_Days": 90}, META)[1]

def test_no_leakage_columns_in_features():
    assert not set(META["features"]) & set(META["excluded_leakage_fields"])
    assert list(pr.build_input_frame(GOOD).columns) == META["features"]

def test_empty_and_malformed_csv():
    assert dp.read_csv_safely(io.StringIO(""))[1]
    assert dp.read_csv_safely(io.StringIO("a,b\n"))[1]

def test_missing_columns():
    df = pd.read_csv(dp.DEFAULT_DATA).drop(columns=["Time_Slot"])
    assert "Time_Slot" in dp.validate_schema(df, META).errors[0]

def test_cleaning_and_scoring():
    raw = pd.read_csv(dp.DEFAULT_DATA)
    dirty = pd.concat([raw, raw.iloc[5:8]]).reset_index(drop=True)
    dirty.loc[0, "Lead_Time_Days"] = -5
    dirty.loc[1, "Facility_Type"] = "Pool"
    clean, log = dp.clean_dataset(dirty, META)
    assert len(clean) == 400 - 1 and (clean["Lead_Time_Days"] >= 0).all()
    s = db.score(MODEL, clean)
    assert set(s["Risk_Level"]) <= {"High", "Medium", "Low"}
