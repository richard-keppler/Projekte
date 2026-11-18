from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.model import load_model
from src.preprocessing import FEATURES, convert_hour_weekday


class TripRequest(BaseModel):
    PULocationID: int = Field(ge=1, le=265)
    DOLocationID: int = Field(ge=1, le=265)
    trip_distance: float = Field(gt=0, le=100)
    tpep_pickup_datetime: datetime


models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["predict_duration"] = load_model()
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_duration(triprequest: TripRequest):
    triprequest_dict = triprequest.model_dump()
    triprequest_df = pd.DataFrame([triprequest_dict])
    triprequest_df = convert_hour_weekday(triprequest_df)
    features = triprequest_df[FEATURES]
    prediction = models["predict_duration"].predict(features)
    return {"predicted_duration_minutes": float(prediction[0])}
