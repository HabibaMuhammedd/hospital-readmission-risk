"""Streamlit discharge form -> 30-day readmission risk.   Run:  streamlit run app.py"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import DIAG_GROUPS, MED_COLS, prepare  # noqa: E402

st.set_page_config(page_title="30-Day Readmission Risk", page_icon="🏥", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(Path(__file__).parent / "models" / "readmission_model.joblib")


bundle = load_bundle()
pipe, threshold, defaults = bundle["pipeline"], bundle["threshold"], bundle["defaults"]
cats = bundle["categories"]

# representative raw ICD-9 code for each diagnosis group (the model only sees the group)
ICD_EXAMPLE = {"Circulatory": "428", "Respiratory": "486", "Digestive": "560", "Diabetes": "250.8",
               "Injury": "820", "Musculoskeletal": "715", "Genitourinary": "599",
               "Neoplasms": "153", "Other": "V57", "Unknown": None}
AGES = [f"[{i}-{i+10})" for i in range(0, 100, 10)]

# friendly labels for the coded fields in the dataset (see IDS_mapping.csv)
A1C_LABELS = {"Unknown": "Not tested", "Norm": "Normal", ">7": "Above 7%", ">8": "Above 8%"}
CHANGE_LABELS = {"Ch": "Medication changed", "No": "No change"}
INSULIN_LABELS = {"No": "Not prescribed", "Steady": "Steady dose", "Up": "Dose increased", "Down": "Dose decreased"}
DISCHARGE = {1: "Home", 6: "Home with home-health service", 3: "Skilled nursing facility",
             2: "Another short-term hospital", 22: "Rehabilitation facility",
             5: "Other inpatient care institution", 4: "Intermediate care facility",
             23: "Long-term care hospital", 7: "Left against medical advice"}
ADMISSION_TYPE = {1: "Emergency", 2: "Urgent", 3: "Elective", 7: "Trauma center"}
ADMISSION_SOURCE = {7: "Emergency room", 1: "Physician referral", 2: "Clinic referral",
                    3: "HMO referral", 4: "Transfer from a hospital",
                    5: "Transfer from a skilled nursing facility", 6: "Transfer from another facility"}


def opts(mapping, key, default):
    """Keep only options the model has seen, and pick the training default."""
    ids = [k for k in mapping if str(k) in cats[key]]
    return ids, ids.index(int(default)) if int(default) in ids else 0

st.title("🏥 30-Day Readmission Risk — Diabetic Patients")
st.caption("Decision-support for planning follow-up. Not a diagnosis. Trained on US hospital data "
           "(1999–2008), so treat results as a demonstration, not a clinical tool.")

with st.form("discharge"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Patient")
        age = st.selectbox("Age group", AGES, index=7)
        gender = st.selectbox("Gender", ["Female", "Male"])
        race = st.selectbox("Race", [r for r in cats["race"]], index=cats["race"].index(defaults["race"]))
    with c2:
        st.subheader("This stay")
        time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
        num_lab = st.slider("Lab procedures", 1, 130, 43)
        num_proc = st.slider("Non-lab procedures", 0, 6, 1)
        num_med = st.slider("Distinct medications", 1, 80, 16)
        num_diag = st.slider("Number of diagnoses", 1, 16, 8)
        diag1 = st.selectbox("Primary diagnosis group", DIAG_GROUPS[:-1])
    with c3:
        st.subheader("History & treatment")
        n_out = st.number_input("Outpatient visits (past year)", 0, 50, 0)
        n_emer = st.number_input("Emergency visits (past year)", 0, 50, 0)
        n_inp = st.number_input("Inpatient visits (past year)", 0, 30, 0)
        a1c = st.selectbox("HbA1c result", cats["A1Cresult"], index=cats["A1Cresult"].index("Unknown"),
                           format_func=lambda v: A1C_LABELS.get(v, v))
        insulin = st.selectbox("Insulin", cats["insulin"], index=cats["insulin"].index("No"),
                               format_func=lambda v: INSULIN_LABELS.get(v, v))
        change = st.selectbox("Diabetes medication change during stay", cats["change"],
                              format_func=lambda v: CHANGE_LABELS.get(v, v))
        dmed = st.selectbox("Diabetes medication prescribed", cats["diabetesMed"],
                            index=cats["diabetesMed"].index("Yes"))
        d_ids, d_def = opts(DISCHARGE, "discharge_disposition_id", defaults["discharge_disposition_id"])
        disp = st.selectbox("Discharged to", d_ids, index=d_def, format_func=DISCHARGE.get)
        t_ids, t_def = opts(ADMISSION_TYPE, "admission_type_id", defaults["admission_type_id"])
        adm_type = st.selectbox("Admission type", t_ids, index=t_def, format_func=ADMISSION_TYPE.get)
        s_ids, s_def = opts(ADMISSION_SOURCE, "admission_source_id", defaults["admission_source_id"])
        adm_src = st.selectbox("Admission source", s_ids, index=s_def, format_func=ADMISSION_SOURCE.get)
    go = st.form_submit_button("Estimate risk", type="primary")


def build_row(**over):
    """Raw-format row: stored defaults for everything the form doesn't ask about."""
    raw = {
        "age": age, "gender": gender, "race": race, "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab, "num_procedures": num_proc, "num_medications": num_med,
        "number_diagnoses": num_diag, "number_outpatient": n_out, "number_emergency": n_emer,
        "number_inpatient": n_inp, "A1Cresult": a1c, "insulin": insulin, "change": change,
        "diabetesMed": dmed, "discharge_disposition_id": int(disp),
        "diag_1": ICD_EXAMPLE[diag1], "diag_2": None, "diag_3": None,
        "admission_type_id": int(adm_type), "admission_source_id": int(adm_src),
        "medical_specialty": None if defaults["medical_specialty"] == "Unknown" else defaults["medical_specialty"],
        "max_glu_serum": None if defaults["max_glu_serum"] == "Unknown" else defaults["max_glu_serum"],
    }
    raw.update({m: "No" for m in MED_COLS})
    raw["insulin"] = insulin
    raw["metformin"] = defaults["metformin"]
    raw.update(over)
    return pd.DataFrame([raw])


def risk(row):
    return float(pipe.predict_proba(prepare(row))[0, 1])


if go:
    row = build_row()
    p = risk(row)
    high = p >= threshold
    a, b, c = st.columns(3)
    a.metric("Estimated 30-day readmission risk", f"{p:.0%}")
    b.metric("Average patient in training data", f"{bundle['metrics']['base_rate']:.0%}")
    c.metric("Follow-up flag", "HIGH — plan follow-up" if high else "Standard")
    (st.error if high else st.success)(
        f"Flag threshold is {threshold:.0%} (chosen to catch ~{bundle['metrics']['target_recall_cv']:.0%} "
        f"of readmissions). Scores are for ranking patients; they are not calibrated probabilities.")

    # model-agnostic explanation: reset one input at a time to a typical value, see how risk moves
    typical = {"time_in_hospital": 3, "num_lab_procedures": 44, "num_procedures": 1,
               "num_medications": 15, "number_diagnoses": 8, "number_outpatient": 0,
               "number_emergency": 0, "number_inpatient": 0}
    labels = {"time_in_hospital": "Days in hospital", "num_lab_procedures": "Lab procedures",
              "num_procedures": "Procedures", "num_medications": "Medications",
              "number_diagnoses": "Number of diagnoses", "number_outpatient": "Outpatient visits",
              "number_emergency": "Emergency visits", "number_inpatient": "Inpatient visits"}
    effects = {labels[k]: p - risk(build_row(**{k: v})) for k, v in typical.items()}
    effects["Primary diagnosis"] = p - risk(build_row(diag_1=None))
    effects["Insulin"] = p - risk(build_row(insulin="No"))
    effects["HbA1c"] = p - risk(build_row(A1Cresult=None))
    effects["Medication change"] = p - risk(build_row(change="No"))
    s = pd.Series(effects).sort_values(ascending=False)
    st.subheader("What moves this patient's risk")
    st.caption("Change in estimated risk versus a typical value for each factor "
               "(positive = pushes risk up).")
    st.bar_chart(s.head(8).round(4) * 100, horizontal=True)

with st.expander("About this model"):
    m = bundle["metrics"]
    st.write(f"**Model:** {bundle['model_name']} · hold-out ROC-AUC {m['test_roc_auc']:.2f} · "
             f"PR-AUC {m['test_pr_auc']:.2f} (base rate {m['base_rate']:.2f}) · "
             f"recall {m['test_recall']:.0%} · precision {m['test_precision']:.0%}")
    st.write("Readmission is hard to predict; modest AUC is expected on this dataset. "
             "Split was by patient, and patients who died or went to hospice were excluded.")
    cv = Path(__file__).parent / "models" / "cv_results.csv"
    if cv.exists():
        st.dataframe(pd.read_csv(cv).round(3), hide_index=True)
