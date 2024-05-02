from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


app = FastAPI()

# Load the trained model
logistic_model = joblib.load("./Models/logistic_model.joblib")
random_forest_model = joblib.load("./Models/random_forest_model.joblib")

# Define input data model
class InputData(BaseModel):
    REGION: str
    TENURE: str
    MONTANT: float
    FREQUENCE_RECH: float
    REVENUE: float
    ARPU_SEGMENT: float
    FREQUENCE: float
    DATA_VOLUME: float
    ON_NET: float
    ORANGE: float
    TIGO: float
    REGULARITY: int
    FREQ_TOP_PACK: float

@app.get('/')
def home():   
    return {
        "title": "Welcome to the CHURN GUARD API!",
        "description": "Predict the likelihood of customer churn in Expresso using machine learning models.",
        "endpoints": {
            "/predict_with_logistic_model": "Predict churn using the Logistic Regression model.",
            "/predict_with_random_forest_model": "Predict churn using the Random Forest model."
        },
        "usage": {
            "method": "POST",
            "content_type": "application/json",
            "data_structure": {
                "REGION": "String",
                "TENURE": "String",
                "MONTANT": "Float",
                "FREQUENCE_RECH": "Float",
                "REVENUE": "Float",
                "ARPU_SEGMENT": "Float",
                "FREQUENCE": "Float",
                "DATA_VOLUME": "Float",
                "ON_NET": "Float",
                "ORANGE": "Float",
                "TIGO": "Float",
                "REGULARITY": "Integer",
                "FREQ_TOP_PACK": "Float"
            },
            "example_data": {
                "REGION": "DAKAR",
                "TENURE": "K > 24 month",
                "MONTANT": 120,
                "FREQUENCE_RECH": 2,
                "REVENUE": 1200,
                "ARPU_SEGMENT": 400,
                "FREQUENCE": 10,
                "DATA_VOLUME": 500,
                "ON_NET": 200,
                "ORANGE": 150,
                "TIGO": 50,
                "REGULARITY": 6,
                "FREQ_TOP_PACK": 5
            }
        }
    }

# Prediction endpoint
@app.post("/predict_with_logistic_model")
def predict_churn(data: InputData):
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.model_dump()])
        
    # Make predictions
    prediction = logistic_model.predict(input_df)
    probability = logistic_model.predict_proba(input_df)

    probability = probability.tolist() # convert the probability result to a list
        
    # Prepare response
    if prediction[0] == 1:
        result = "Churn"
    else:
        result = "No Churn"
        
    return {"prediction": result, "probability": probability}
    
# Prediction endpoint
@app.post("/predict_with_random_forest_model")
def predict_churn(data: InputData):

    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.model_dump()])
        
    # Make predictions
    prediction = random_forest_model.predict(input_df)
    probability = random_forest_model.predict_proba(input_df)

    probability = probability.tolist() # convert the probability result to a list
        
    # Prepare response
    if prediction[0] == 1:
        result = "Churn"
    else:
        result = "No Churn"
        
    return {"prediction": result, "probability": probability}

# Run the FastAPI app
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)