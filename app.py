
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("model.joblib")


class HeartData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API"}


@app.post("/predict")
def predict(data: HeartData):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = int(model.predict(input_data)[0])

    return {
        "prediction": prediction,
        "result": (
            "Heart disease recorded"
            if prediction == 1
            else "No heart disease recorded"
        )
    }
