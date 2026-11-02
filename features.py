"""Cleaning + feature engineering shared by training and the Streamlit app."""
import numpy as np
import pandas as pd

# discharge_disposition_id values meaning the patient died or went to hospice
# (cannot be readmitted -> remove them, otherwise they look like "no readmission")
EXCLUDE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

MED_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "examide", "citoglipton", "insulin", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone"]

NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
           "age_mid", "prior_visits", "n_med_changes", "n_meds_on"]
CATEGORICAL = ["race", "gender", "admission_type_id", "discharge_disposition_id",
               "admission_source_id", "medical_specialty", "diag_1_grp", "diag_2_grp",
               "diag_3_grp", "max_glu_serum", "A1Cresult", "insulin", "metformin",
               "change", "diabetesMed"]
FEATURES = NUMERIC + CATEGORICAL
DIAG_GROUPS = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
               "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Unknown"]


def icd9_group(code):
    """Group an ICD-9 code into a clinical category (Strack et al. 2014 style)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Unknown"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    try:
        n = float(code)
    except ValueError:
        return "Other"
    if 250 <= n < 251:
        return "Diabetes"
    if 390 <= n <= 459 or n == 785:
        return "Circulatory"
    if 460 <= n <= 519 or n == 786:
        return "Respiratory"
    if 520 <= n <= 579 or n == 787:
        return "Digestive"
    if 800 <= n <= 999:
        return "Injury"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 580 <= n <= 629 or n == 788:
        return "Genitourinary"
    if 140 <= n <= 239:
        return "Neoplasms"
    return "Other"


def load_raw(path):
    return pd.read_csv(path, na_values="?", low_memory=False)


def clean_training_frame(df):
    """Row-level filters that only make sense for training data."""
    df = df[~df["discharge_disposition_id"].isin(EXCLUDE_DISPOSITIONS)]
    df = df[df["gender"].isin(["Male", "Female"])]
    return df.reset_index(drop=True)


def make_target(df):
    return (df["readmitted"] == "<30").astype(int)


def prepare(df):
    """Raw rows (original column names) -> model-ready feature frame."""
    d = pd.DataFrame(index=df.index)
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]:
        d[c] = pd.to_numeric(df[c])
    d["age_mid"] = df["age"].astype(str).str.extract(r"\[(\d+)-")[0].astype(float) + 5
    d["prior_visits"] = d["number_outpatient"] + d["number_emergency"] + d["number_inpatient"]
    meds = df[MED_COLS]
    d["n_med_changes"] = meds.isin(["Up", "Down"]).sum(axis=1)
    d["n_meds_on"] = (meds != "No").sum(axis=1)
    for c in ["race", "gender", "max_glu_serum", "A1Cresult", "insulin", "metformin",
              "change", "diabetesMed", "medical_specialty"]:
        d[c] = df[c].fillna("Unknown").astype(str)
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        d[c] = df[c].astype(int).astype(str)   # IDs are categories, not quantities
    for i in (1, 2, 3):
        d[f"diag_{i}_grp"] = df[f"diag_{i}"].map(icd9_group)
    return d[FEATURES]
