from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "salaries.csv"

EDUCATION_NORMALIZATION = {
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    "phD": "PhD",
}

JOB_TITLE_NORMALIZATION = {
    "Back end Developer": "Back End Developer",
    "Front end Developer": "Front End Developer",
    "Juniour HR Coordinator": "Junior HR Coordinator",
    "Juniour HR Generalist": "Junior HR Generalist",
}

ALLOWED_GENDERS = ["Female", "Male"]


def load_clean_dataset():
    dataframe = pd.read_csv(DATASET_PATH).dropna().copy()
    dataframe["Education Level"] = dataframe["Education Level"].replace(
        EDUCATION_NORMALIZATION
    )
    dataframe["Job Title"] = dataframe["Job Title"].replace(JOB_TITLE_NORMALIZATION)
    return dataframe


def get_form_options():
    dataframe = load_clean_dataset()

    return {
        "genders": ALLOWED_GENDERS,
        "education_levels": sorted(dataframe["Education Level"].unique().tolist()),
        "job_titles": sorted(dataframe["Job Title"].unique().tolist()),
        "age_min": int(dataframe["Age"].min()),
        "age_max": int(dataframe["Age"].max()),
        "experience_min": float(dataframe["Years of Experience"].min()),
        "experience_max": float(dataframe["Years of Experience"].max()),
    }
