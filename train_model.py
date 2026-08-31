"""
train_model.py
Generates a synthetic loan-approval dataset and trains a RandomForest
classifier to predict whether a loan should be approved.

Run this once locally: python train_model.py
It produces model.pkl, which the API (app.py) loads at request time.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# 1. Generate synthetic dataset
# -----------------------------
np.random.seed(42)
n_samples = 3000

income = np.random.normal(loc=250000, scale=90000, size=n_samples).clip(30000)
credit_score = np.random.normal(loc=650, scale=90, size=n_samples).clip(300, 850)
loan_amount = np.random.normal(loc=800000, scale=350000, size=n_samples).clip(50000)
employment_years = np.random.exponential(scale=4, size=n_samples).clip(0, 35)
existing_debt = np.random.normal(loc=150000, scale=100000, size=n_samples).clip(0)

# A simple underlying "rule" that approval depends on, plus noise,
# so the model has a real signal to learn from.
debt_to_income = existing_debt / income
loan_to_income = loan_amount / income

approval_score = (
    (credit_score - 600) / 150
    - loan_to_income * 0.8
    - debt_to_income * 0.6
    + (employment_years / 15)
    + np.random.normal(0, 0.5, n_samples)
)

# Threshold on the median so the classes come out reasonably balanced
threshold = np.median(approval_score)
approved = (approval_score > threshold).astype(int)

df = pd.DataFrame({
    "income": income,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "employment_years": employment_years,
    "existing_debt": existing_debt,
    "approved": approved,
})

print("Dataset preview:")
print(df.head())
print(f"\nApproval rate: {df['approved'].mean():.2%}")

# -----------------------------
# 2. Train / test split
# -----------------------------
FEATURES = ["income", "credit_score", "loan_amount", "employment_years", "existing_debt"]
X = df[FEATURES]
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
)
model.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate
# -----------------------------
preds = model.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, preds):.3f}")
print("\nClassification report:")
print(classification_report(y_test, preds))

# -----------------------------
# 5. Save model to disk
# -----------------------------
joblib.dump({"model": model, "features": FEATURES}, "model.pkl")
print("\nSaved trained model to model.pkl")
