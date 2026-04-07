from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_utils import load_clean_dataset

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "static" / "plots"


def generate_visualizations(dataframe):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("ggplot")

    avg_salary_by_education = (
        dataframe.groupby("Education Level")["Salary"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    avg_salary_by_education.plot(kind="bar", color="#2a6f97")
    plt.title("Average Salary by Education Level")
    plt.xlabel("Education Level")
    plt.ylabel("Average Salary")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "salary_by_education.png")
    plt.close()

    gender_distribution = dataframe["Gender"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(
        gender_distribution,
        labels=gender_distribution.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#468faf", "#f28482", "#84a98c", "#f6bd60"],
    )
    plt.title("Gender Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "gender_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        dataframe["Years of Experience"],
        dataframe["Salary"],
        alpha=0.6,
        color="#bc4749",
        edgecolors="white",
        linewidths=0.5,
    )
    plt.title("Salary vs Years of Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "experience_vs_salary.png")
    plt.close()


# Load dataset
df = load_clean_dataset()

# Generate charts from human-readable data before encoding
generate_visualizations(df.copy())

# Encode categorical columns
le_gender = LabelEncoder()
le_education = LabelEncoder()
le_job = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Education Level"] = le_education.fit_transform(df["Education Level"])
df["Job Title"] = le_job.fit_transform(df["Job Title"])

# Features & target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

# Save model + encoders
pickle.dump(model, open(BASE_DIR / "model.pkl", "wb"))
pickle.dump(le_gender, open(BASE_DIR / "le_gender.pkl", "wb"))
pickle.dump(le_education, open(BASE_DIR / "le_education.pkl", "wb"))
pickle.dump(le_job, open(BASE_DIR / "le_job.pkl", "wb"))

print("Model trained and saved successfully!")
print(f"R2 score: {score:.4f}")
print(f"Charts saved in: {PLOTS_DIR}")
