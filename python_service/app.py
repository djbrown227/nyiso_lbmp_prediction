from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import List

# Load the trained model
model = joblib.load('/Users/danielbrown/Desktop/WebApps/NYISO_PyRustGo/python_service/best_model.pkl')

# Initialize FastAPI
app = FastAPI()

# Define a data model for the input
class PricePredictionInput(BaseModel):
    lbmp: float
    marginal_cost_losses: float
    marginal_cost_congestion: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "NYISO Price Prediction API"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: List[PricePredictionInput]):
    # Prepare the input data for prediction
    input_data = pd.DataFrame([{
        'LBMP ($/MWHr)': item.lbmp,
        'Marginal Cost Losses ($/MWHr)': item.marginal_cost_losses,
        'Marginal Cost Congestion ($/MWHr)': item.marginal_cost_congestion
    } for item in data])

    # Perform prediction
    try:
        prediction = model.predict(input_data)
        return {"predictions": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# To run the API:
# uvicorn app:app --host 127.0.0.1 --port 8000 --reload
