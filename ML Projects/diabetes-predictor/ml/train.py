# ml/train.py
# Train NB & Logistic Regression on PIMA, evaluate, plot, and export best model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, RocCurveDisplay, precision_recall_fscore_support
)
import joblib

DATA_PATH = Path("ml") / "diabetes.csv"  # place Kaggle CSV here (Pima Indians Diabetes Database)
ARTIFACTS = Path("ml") / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------- 1. LOAD ----------
df = pd.read_csv(DATA_PATH)

# Columns where zeros are invalid measurements
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# ---------- 1. CLEAN ----------
df_clean = df.copy()
for col in zero_as_missing:
    # Replace 0 with NaN then fill with column mean
    df_clean[col] = df_clean[col].replace(0, np.nan)
    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

# Ensure correct dtypes (floats)
for col in df_clean.columns:
    if col != "Outcome":
        df_clean[col] = df_clean[col].astype(float)

# ---------- 2. EDA ----------
print("\n=== Basic Stats ===")
print(df_clean.describe())

# Histograms
for col in df_clean.columns.drop("Outcome"):
    plt.figure()
    df_clean[col].hist(bins=30)
    plt.title(f"Distribution: {col}")
    plt.xlabel(col); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / f"hist_{col}.png")
    plt.close()

# Correlation heatmap
import seaborn as sns  # only used for heatmap image here; not required for training
plt.figure(figsize=(8,6))
sns.heatmap(df_clean.corr(), annot=False, cmap="coolwarm", linewidths=.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(ARTIFACTS / "correlation_heatmap.png")
plt.close()

# ---------- 3. FEATURE/TARGET ----------
X = df_clean.drop(columns=["Outcome"])
y = df_clean["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 4. MODELS ----------
pipe_nb = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GaussianNB())
])

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

models = {
    "NaiveBayes": pipe_nb,
    "LogReg": pipe_lr
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    auc = roc_auc_score(y_test, proba)

    results[name] = dict(acc=acc, cm=cm, precision=pr, recall=rc, f1=f1, auc=auc)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f} | Precision: {pr:.4f} | Recall: {rc:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, preds, zero_division=0))

# ---------- 5. ROC CURVES ----------
plt.figure()
for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)
plt.title("ROC Curves")
plt.tight_layout()
plt.savefig(ARTIFACTS / "roc_curves.png")
plt.close()

# Pick best by AUC (or accuracy as tie-breaker)
best_name = sorted(results.items(), key=lambda kv: (kv[1]['auc'], kv[1]['acc']), reverse=True)[0][0]
best_model = models[best_name]

print(f"\n>>> Best model: {best_name} (AUC={results[best_name]['auc']:.4f}, ACC={results[best_name]['acc']:.4f})")

# Save pipeline + metadata
export = {
    "model": best_model,
    "features": list(X.columns),
    "stats": results,
    "best_name": best_name
}
joblib.dump(export, ARTIFACTS / "diabetes_model.joblib")
print(f"Saved model to {ARTIFACTS/'diabetes_model.joblib'}")
