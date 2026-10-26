# Deployment Report - Facility Reservation Monitoring

**Student / Section:** ______  **GitHub repo:** ______  **Streamlit URL:** ______

## 1. Reused code
- Midterm Part II cleaning (duplicates, negative/zero values -> median, derived Lead_Time_Category,
  Reservation_Outcome, Attendance_Ratio) -> `modules/data_processing.py::clean_dataset`.
- Midterm Part III EDA questions (facility volume, outcome by lead-time and user category) -> `modules/exploratory_analysis.py`.
- Midterm Part IV model (same 6 features, 80/20 stratified split, random_state 42,
  DecisionTree max_depth 4) -> `train_model.py`.

## 2. Changes
- LabelEncoder dict -> OrdinalEncoder in a `Pipeline`; saved once with joblib, loaded with `@st.cache_resource`.
- Input validation (lead time >= 0, participants > 0, known categories, out-of-range warning).
- Added risk levels (High >= 50%, Medium 40-50%, Low < 40% of P(Not Completed)), review queue,
  role-gated decision log with reviewer, note, timestamp and model version.
- Midterm notebook bug noted: `fillna(..., inplace=True)` on a column slice is deprecated in recent pandas;
  the app uses assignment instead.

## 3. Tests (see `tests/test_app_logic.py`, plus manual checks in the browser)
| Test | Expected | Result |
|---|---|---|
| Pipeline vs midterm model on 80 test rows | 80/80 identical | Passed (script run) |
| Valid input prediction | label + probability | Passed |
| Negative lead time / 0 participants / unknown facility / NaN | error, no prediction | Passed |
| Lead time 90 (outside 0-20) | warning | Passed |
| Empty CSV, header-only CSV | friendly error | Passed |
| Missing column | error naming the column | Passed |
| Duplicates, invalid values, unknown category cleaning | cleaned / excluded | Passed |
| Streamlit pages render, charts, upload widget, deployed URL | no errors | **Fill in after you run the app** |

## 4. Limitations
1. Accuracy 55.0% < majority baseline 57.5%; recall for Not Completed is 26% (9 of 34).
2. Synthetic, randomly generated data (400 rows) - patterns are weak and may be noise.
3. Probabilities are tree-leaf proportions, not calibrated.
4. Roles are simulated; session logs reset when the app restarts.
5. Dashboard scores historical rows to demonstrate the workflow; in production it would score upcoming reservations.

## 5. Reflection
(Write 3-5 sentences: what you learned about moving a model from notebook to information system,
why human review matters given the weak model, and what you would improve with real data.)
