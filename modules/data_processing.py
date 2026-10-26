"""Dataset loading, schema validation, quality summary and cleaning.

Pure pandas (no Streamlit import) so it can be unit-tested. The cleaning rules are
the ones used in Part II of the Midterm notebook.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = BASE_DIR / "data" / "cleaned_facility_reservations.csv"
META_PATH = BASE_DIR / "models" / "model_metadata.json"

FEATURE_COLUMNS = ["User_Category", "Facility_Type", "Lead_Time_Days",
                   "Participants", "Time_Slot", "Day_of_Week"]
CATEGORICAL = ["User_Category", "Facility_Type", "Time_Slot", "Day_of_Week"]
NUMERIC = ["Lead_Time_Days", "Participants"]
OPTIONAL_COLUMNS = ["Reservation_ID", "Usage_Status", "Reservation_Outcome",
                    "Actual_Attendance", "Attendance_Ratio", "Lead_Time_Category"]
MAX_UPLOAD_MB = 10


def load_metadata() -> dict:
    with open(META_PATH) as f:
        return json.load(f)


@dataclass
class ValidationResult:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_csv_safely(uploaded) -> tuple[pd.DataFrame | None, str | None]:
    """Return (dataframe, error_message). Never raises for bad user files."""
    try:
        size_mb = getattr(uploaded, "size", 0) / 1_000_000
        if size_mb > MAX_UPLOAD_MB:
            return None, f"File is {size_mb:.1f} MB; the limit is {MAX_UPLOAD_MB} MB."
        df = pd.read_csv(uploaded)
    except pd.errors.EmptyDataError:
        return None, "The file is empty - no columns could be read."
    except pd.errors.ParserError as e:
        return None, f"The file is not a valid CSV: {e}"
    except UnicodeDecodeError:
        return None, "The file could not be decoded as text. Save it as UTF-8 CSV."
    except Exception as e:  # noqa: BLE001 - show a friendly message instead of a traceback
        return None, f"Could not read the file: {e}"
    if df.empty:
        return None, "The file has headers but no data rows."
    df.columns = [str(c).strip() for c in df.columns]
    return df, None


def validate_schema(df: pd.DataFrame, meta: dict | None = None) -> ValidationResult:
    meta = meta or load_metadata()
    res = ValidationResult()
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        res.errors.append(
            "Missing required column(s): " + ", ".join(missing) +
            ". Required: " + ", ".join(FEATURE_COLUMNS) + ".")
        return res
    if "Reservation_Outcome" not in df.columns and "Usage_Status" not in df.columns:
        res.warnings.append("No Reservation_Outcome / Usage_Status column: EDA on outcomes "
                            "and model-vs-actual comparisons will be unavailable.")
    for col in NUMERIC:
        bad = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
        if bad.any():
            res.warnings.append(f"{col}: {int(bad.sum())} non-numeric value(s) will be treated as missing.")
    for col in CATEGORICAL:
        vals = df[col].dropna().astype(str).str.strip()
        unknown = sorted(set(vals) - set(meta["allowed_categories"][col]))
        if unknown:
            res.warnings.append(
                f"{col}: unknown value(s) {unknown}. Rows with these values cannot be scored "
                f"and will be excluded. Allowed: {meta['allowed_categories'][col]}.")
    return res


def quality_summary(df: pd.DataFrame) -> dict:
    nulls = df.isna().sum()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_nulls": int(nulls.sum()),
        "nulls_by_column": nulls[nulls > 0].to_dict(),
    }


def categorize_lead_time(days: float) -> str:
    if days <= 2:
        return "Last Minute"
    if days <= 7:
        return "Short Lead"
    if days <= 14:
        return "Medium Lead"
    return "Early Booking"


def clean_dataset(df: pd.DataFrame, meta: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Apply the midterm cleaning rules. Returns (clean_df, log_messages)."""
    meta = meta or load_metadata()
    log: list[str] = []
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    n0 = len(out)
    out = out.drop_duplicates()
    log.append(f"Removed {n0 - len(out)} duplicate row(s).")

    for col in CATEGORICAL:
        out[col] = out[col].astype("string").str.strip()
    known = pd.Series(True, index=out.index)
    for col in CATEGORICAL:
        known &= out[col].isin(meta["allowed_categories"][col])
    if (~known).any():
        log.append(f"Excluded {int((~known).sum())} row(s) with missing/unknown categories.")
    out = out[known].copy()

    for col, invalid in (("Lead_Time_Days", lambda s: s < 0), ("Participants", lambda s: s <= 0)):
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[invalid(out[col]), col] = np.nan
        med = out[col].median()
        if pd.isna(med):
            med = meta["training_medians"][col]
        n_na = int(out[col].isna().sum())
        out[col] = out[col].fillna(med).astype(float)
        log.append(f"{col}: {n_na} missing/invalid value(s) replaced by median ({med:g}).")

    if "Actual_Attendance" in out.columns:
        out["Actual_Attendance"] = pd.to_numeric(out["Actual_Attendance"], errors="coerce").fillna(0)
    out["Lead_Time_Category"] = out["Lead_Time_Days"].apply(categorize_lead_time)
    if "Usage_Status" in out.columns:
        out["Reservation_Outcome"] = out["Usage_Status"].apply(
            lambda x: "Completed" if str(x).strip().lower() == "used" else "Not Completed")
    if "Actual_Attendance" in out.columns:
        out["Attendance_Ratio"] = np.where(
            out["Participants"] > 0, out["Actual_Attendance"] / out["Participants"], 0.0)
        out["Attendance_Ratio"] = out["Attendance_Ratio"].clip(upper=1.0)
    return out.reset_index(drop=True), log


def has_outcome(df: pd.DataFrame) -> bool:
    return "Reservation_Outcome" in df.columns
