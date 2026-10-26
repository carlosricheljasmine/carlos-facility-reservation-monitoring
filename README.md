# Campus Facility Reservation Monitoring (Streamlit)

Deployment of the Midterm Laboratory Examination (Section A): predict whether a facility
reservation will be **Completed** or **Not Completed** (cancellation / no-show) and support the
**Facility Administrator's** review. Predictions are advisory; a human records the final decision.

## Pages
Home | Dataset (upload + validation) | Data Analysis (EDA, 4 interactive chart tabs) |
Prediction (single prediction, metrics, consistency check, limitations) |
Dashboard (KPIs, facility risk, review queue, decision log) | System Design (FR/NFR, use case, workflow)

## Reused from the midterm
| Item | Value |
|---|---|
| Dataset | `data/cleaned_facility_reservations.csv` (400 synthetic rows, output of midterm Part II) |
| Target | `Reservation_Outcome` (Completed / Not Completed) |
| Features | User_Category, Facility_Type, Lead_Time_Days, Participants, Time_Slot, Day_of_Week |
| Excluded (leakage) | Usage_Status, Actual_Attendance, Attendance_Ratio, Reservation_Outcome |
| Model | `DecisionTreeClassifier(max_depth=4, random_state=42)`, 80/20 stratified split |

Changed: LabelEncoders -> `OrdinalEncoder` inside one scikit-learn `Pipeline` saved with joblib.
`train_model.py` asserts the pipeline predicts identically to the midterm method on all test rows.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py          # optional: regenerates models/ (already committed)
python make_diagrams.py        # optional: regenerates assets/ (needs graphviz)
streamlit run app.py
python -m pytest tests -q      # needs pytest
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to a **public GitHub repo** (no secrets or real student data).
2. https://share.streamlit.io -> *Create app* -> pick repo/branch -> main file `app.py`.
3. In *Advanced settings* choose Python 3.11 or 3.12 (scikit-learn 1.8 needs >= 3.11).
4. If the model fails to load after deploy, the scikit-learn version differs from the training one:
   run `python train_model.py` with the pinned version, commit `models/`, and redeploy.

## Known limitations
Test accuracy 55.0% is below the 57.5% majority-class baseline; data are synthetic; roles are
simulated (no real authentication). Do not use scores to deny services.
