"""Train the model and save scaler + model together as one pipeline.

Usage: python train.py path/to/star_classification.csv
(Dataset: SDSS17 "Stellar Classification" on Kaggle)
"""
import sys
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

FEATURES = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]

data = pd.read_csv(sys.argv[1])
enc = LabelEncoder()
y = enc.fit_transform(data["class"])  # GALAXY=0, QSO=1, STAR=2
X = data[FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE on the training split only, so the test set stays untouched
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

pipe = Pipeline([("scaler", StandardScaler()), ("model", XGBClassifier(random_state=0))])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=enc.classes_))

dump({"pipeline": pipe, "classes": list(enc.classes_), "features": FEATURES}, "stellar_pipeline.joblib")
print("Saved stellar_pipeline.joblib")
