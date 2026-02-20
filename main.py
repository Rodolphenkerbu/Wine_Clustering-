from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load your model (adjust filename if needed)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"status": "Model service is running"}

@app.post("/predict")
def predict(features: list):
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)
    return {"prediction": pred.tolist()}
