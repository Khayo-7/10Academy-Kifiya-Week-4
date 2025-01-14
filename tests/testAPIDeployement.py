import pytest
import requests
import pandas as pd
from fastapi.testclient import TestClient
from deployment.app.main import app

client = TestClient(app)

def load_data(data=None):
   
    if not data:
        data = pd.DataFrame({
            "Store": [1],
            "DayOfWeek": [5],
            "Promo": [1],
            "SchoolHoliday": [0],
            "StateHoliday": [0]
        })
    payload = {
        "columns": data.columns.tolist(),
        "data": data.values.tolist(),
    }

    return payload

def testPredictSuccess():

    payload = load_data()
    response = client.post("/predict_sales", json=payload)
    print(response.json())

    assert response.status_code == 200
    assert "prediction" in response.json()

def testDeployementSuccess():

    payload = load_data()
    response = requests.post("http://127.0.0.1:1234/invocations", json=payload)

    print(response.json())
    assert response.status_code == 200
    assert "PredictedSales" in response.json()