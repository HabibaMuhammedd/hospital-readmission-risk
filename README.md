# 30-Day Hospital Readmission Risk (Diabetic Patients)

Predicts which diabetic patients are likely to be readmitted within 30 days of discharge, so a care
team can prioritise follow-up calls and visits. Includes a Streamlit discharge form.

**Live demo:** _add your Streamlit link here_

## Data
[Diabetes 130-US hospitals for years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
(101,766 encounters). Download `diabetic_data.csv` into `data/` (not committed to the repo).

## Approach
- Target: readmitted `<30` days (≈11% positive).
- Excluded patients who died or went to hospice (they cannot be readmitted).
- `?` treated as missing; ICD-9 codes grouped into clinical categories; prior-visit and
  medication-change counts engineered.
- **Split by patient** (GroupShuffleSplit / GroupKFold) so the same patient never appears in both train and test.
- Compared Logistic Regression, Random Forest, HistGradientBoosting (+ XGBoost if installed) with 5-fold grouped CV.
- Threshold picked from out-of-fold predictions to reach ~60% recall (recall-first, as missing a high-risk patient is costlier than an extra call).

## Results
| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
| HistGradientBoosting | 0.669 | 0.220 |
| Random Forest | 0.670 | 0.216 |
| Logistic Regression | 0.662 | 0.211 |

Hold-out: ROC-AUC ≈ 0.68, recall ≈ 59%, precision ≈ 18% (base rate 11%).
Readmission is intrinsically hard to predict from administrative data, so modest AUC is expected.

## Run
```
pip install -r requirements-train.txt
python src/train.py                 # trains, cross-validates, saves models/readmission_model.joblib
python -m streamlit run app.py
```

## Limitations
Decision-support demo, not a clinical tool. US data from 1999–2008. Scores rank patients but are not calibrated probabilities.
