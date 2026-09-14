
from fastapi import FastAPI
import joblib
import datetime

app = FastAPI()

model = joblib.load('ml/models/footfall_model.joblib')

@app.get("/predict")
def predict_next_day():
    tomorrow = datetime.date.today().timetuple().tm_yday + 1
    pred = model.predict([[tomorrow]])
    return {"predicted_quantity": round(pred[0])}
