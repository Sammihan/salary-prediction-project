import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("salaries.csv")

# ---------------- PREPROCESSING ----------------

# Drop missing values
df = df.dropna()

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model + encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_gender, open("le_gender.pkl", "wb"))
pickle.dump(le_education, open("le_education.pkl", "wb"))
pickle.dump(le_job, open("le_job.pkl", "wb"))

print("Model trained and saved successfully!")
