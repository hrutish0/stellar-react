# Stellar Object Classifier (React + FastAPI)

Classifies SDSS observations as Star, Galaxy or Quasar using an XGBoost model.

- **Frontend:** React (Vite)
- **Backend:** FastAPI, serving the model via `POST /predict`
- **ML:** Pandas, Scikit-learn, XGBoost, SMOTE

## Run

Backend
```
cd backend
pip install -r requirements.txt
python train.py path/to/star_classification.csv   # creates stellar_pipeline.joblib
uvicorn main:app --reload
```

Frontend
```
cd frontend
npm install
npm run dev
```
Open http://localhost:5173. API docs at http://localhost:8000/docs.
