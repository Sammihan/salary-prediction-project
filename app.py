from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model & encoders
model = pickle.load(open("model.pkl", "rb"))
le_gender = pickle.load(open("le_gender.pkl", "rb"))
le_education = pickle.load(open("le_education.pkl", "rb"))
le_job = pickle.load(open("le_job.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    gender = request.form["gender"]
    education = request.form["education"]
    job = request.form["job"]
    experience = float(request.form["experience"])

    # Encode inputs
    gender = le_gender.transform([gender])[0]
    education = le_education.transform([education])[0]
    job = le_job.transform([job])[0]

    features = np.array([[age, gender, education, job, experience]])

    prediction = model.predict(features)[0]

    return render_template("result.html", salary=round(prediction, 2))


if __name__ == "__main__":
    app.run(debug=True)
