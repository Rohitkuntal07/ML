# ml/predict.py
# Read JSON from stdin, load exported pipeline, return JSON with prediction & probability.

import sys, json, joblib
from pathlib import Path
import numpy as np
import pandas as pd

ARTIFACT = Path("ml/artifacts/diabetes_model.joblib")
bundle = joblib.load(ARTIFACT)
pipe = bundle["model"]
features = bundle["features"]

# Read one JSON object from stdin
data = json.loads(sys.stdin.read())

# Make a single-row DataFrame in correct column order
# Make a single-row DataFrame in correct column order
X = pd.DataFrame([[float(data.get(f)) for f in features]], columns=features)

proba = pipe.predict_proba(X)[0,1]
pred = int(proba >= 0.5)  # threshold can be tuned

label = "Diabetic - High Risk" if pred == 1 else "Non-Diabetic"


print(json.dumps({
    "prediction": pred,
    "label": label,
    "probability": float(proba),
    "model": bundle["best_name"]
}))
