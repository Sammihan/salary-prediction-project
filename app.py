from pathlib import Path
from flask import Flask, render_template, request
import pickle
import pandas as pd

from data_utils import get_form_options

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Load model & encoders
model = pickle.load(open(BASE_DIR / "model.pkl", "rb"))
le_gender = pickle.load(open(BASE_DIR / "le_gender.pkl", "rb"))
le_education = pickle.load(open(BASE_DIR / "le_education.pkl", "rb"))
le_job = pickle.load(open(BASE_DIR / "le_job.pkl", "rb"))


def get_chart_paths():
    chart_names = [
        "salary_by_education.png",
        "gender_distribution.png",
        "experience_vs_salary.png",
    ]
    return [
        f"plots/{chart_name}"
        for chart_name in chart_names
        if (BASE_DIR / "static" / "plots" / chart_name).exists()
    ]


def render_home(error=None, form_data=None):
    options = get_form_options()
    return render_template(
        "index.html",
        charts=get_chart_paths(),
        error=error,
        form_data=form_data or {},
        genders=options["genders"],
        education_levels=options["education_levels"],
        job_titles=options["job_titles"],
        age_min=options["age_min"],
        age_max=options["age_max"],
        experience_min=options["experience_min"],
        experience_max=options["experience_max"],
    )


@app.route("/")
def home():
    return render_home()


@app.route("/predict", methods=["POST"])
def predict():
    options = get_form_options()
    form_data = {
        "age": request.form.get("age", "").strip(),
        "gender": request.form.get("gender", "").strip(),
        "education": request.form.get("education", "").strip(),
        "job": request.form.get("job", "").strip(),
        "experience": request.form.get("experience", "").strip(),
    }

    try:
        age = int(form_data["age"])
    except ValueError:
        return render_home("Age must be a whole number.", form_data)

    try:
        experience = float(form_data["experience"])
    except ValueError:
        return render_home("Years of experience must be a valid number.", form_data)

    if not options["age_min"] <= age <= options["age_max"]:
        return render_home(
            f"Age must be between {options['age_min']} and {options['age_max']}.",
            form_data,
        )

    if not options["experience_min"] <= experience <= options["experience_max"]:
        return render_home(
            "Years of experience must be between "
            f"{options['experience_min']:.0f} and {options['experience_max']:.0f}.",
            form_data,
        )

    if form_data["gender"] not in options["genders"]:
        return render_home("Please choose either Male or Female.", form_data)

    if form_data["education"] not in options["education_levels"]:
        return render_home("Please choose a valid education level.", form_data)

    if form_data["job"] not in options["job_titles"]:
        return render_home("Please choose a valid job title.", form_data)

    gender = form_data["gender"]
    education = form_data["education"]
    job = form_data["job"]

    # Encode inputs
    gender = le_gender.transform([gender])[0]
    education = le_education.transform([education])[0]
    job = le_job.transform([job])[0]

    features = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Education Level": education,
                "Job Title": job,
                "Years of Experience": experience,
            }
        ]
    )

    prediction = model.predict(features)[0]

    return render_template("result.html", salary=round(prediction, 2))


if __name__ == "__main__":
    app.run(debug=True)
