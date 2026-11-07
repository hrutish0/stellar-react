from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel

BASE = Path(__file__).parent
FEATURES = ["z", "cam_col", "i", "delta", "MJD", "plate", "spec_obj_ID", "alpha", "field_ID"]
CLASSES = ["GALAXY", "QSO", "STAR"]  # LabelEncoder order (alphabetical)

# Prefer the pipeline from train.py (scaler + model). Fall back to the old model file.
pipe_path = BASE / "stellar_pipeline.joblib"
if pipe_path.exists():
    bundle = load(pipe_path)
    model, CLASSES, FEATURES = bundle["pipeline"], bundle["classes"], bundle["features"]
    SCALED_OK = True
else:
    model = load(BASE / "stellar_model.joblib")
    SCALED_OK = False  # old model was trained on scaled data but the scaler was never saved

app = FastAPI(title="Stellar Object Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StellarInput(BaseModel):
    z: float
    cam_col: int
    i: float
    delta: float
    MJD: float
    plate: float
    spec_obj_ID: float
    alpha: float
    field_ID: float


@app.get("/health")
def health():
    return {"status": "ok", "scaler_included": SCALED_OK}


@app.post("/predict")
def predict(payload: StellarInput):
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    probs = model.predict_proba(row)[0]
    idx = int(probs.argmax())
    return {
        "prediction": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
        "warning": None if SCALED_OK else "Model loaded without scaler; run train.py for reliable results.",
    }
