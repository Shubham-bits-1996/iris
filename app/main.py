# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

import numpy as np

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisResponse(BaseModel):
    predicted_class: int
    class_name: str
    probabilities: dict

app = FastAPI(title="Iris Classification API")

# Load the trained model
# Ensure the model is saved at 'model/best_iris_model.pkl'
model = joblib.load("model/best_iris_model.pkl")
target_names = ["setosa", "versicolor", "virginica"]

@app.post("/predict", response_model=IrisResponse)
def predict(request: IrisRequest):
    # Convert input to feature array
    features = np.array([[request.sepal_length,
                          request.sepal_width,
                          request.petal_length,
                          request.petal_width]])

    # Perform prediction and probability estimation
    preds = model.predict(features)
    probs = model.predict_proba(features)[0]
    idx = int(preds[0])

    # Return structured response
    return IrisResponse(
        predicted_class=idx,
        class_name=target_names[idx],
        probabilities={target_names[i]: float(probs[i]) for i in range(len(target_names))}
    )