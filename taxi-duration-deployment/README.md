# NYC Taxi Trip Duration – Training, Tracking & Deployment

An end-to-end ML project: a model that predicts the duration of NYC Yellow Taxi trips is trained on
~3.2 million trips, tracked and tuned with **MLflow** and **Optuna**, served through a **FastAPI**
service, packaged with **Docker** and deployed publicly on **Render**.
Built as part of my Machine Learning Engineering training.

**Skills shown:** end-to-end ML workflow · experiment tracking (MLflow) · hyperparameter tuning (Optuna) ·
REST API with input validation (FastAPI, Pydantic) · containerisation (Docker) · cloud deployment (Render) ·
training/serving consistency · trade-off between accuracy and model size

```
NYC TLC data ──▶ clean & build features ──▶ train + baseline ──▶ MLflow (runs, Optuna trials, models)
                                                                        │
                                       export model ◀───────────────────┘
                                             │
                                             ▼
                         FastAPI /predict ──▶ Docker image ──▶ Render (public URL)
```

## Live Demo

- **Interactive API docs:** https://taxi-duration-api.onrender.com/docs
- The free instance sleeps after ~15 minutes without traffic; the first request can take up to a minute.

```bash
curl -X POST https://taxi-duration-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID": 142, "DOLocationID": 236, "trip_distance": 3.2, "tpep_pickup_datetime": "2025-01-10T08:00:00"}'
```

The response contains the predicted duration in minutes, e.g. `{"predicted_duration_minutes": 13.9}`.
Invalid input (e.g. an unknown zone ID or a negative distance) returns HTTP `422` with a description of the problem.

## Results

| Model | Test RMSE (minutes) | Model size |
|---|---|---|
| Baseline (always predicts the training mean) | 9.77 | – |
| **Random Forest (deployed)** | **4.39** | 6.6 MB |
| Random Forest tuned with Optuna (20 trials) | 4.06 | 113 MB |

- The deployed model cuts the error by **55 %** compared to the baseline.
- Optuna found a configuration that is **7.6 % more accurate** (`max_depth=19`, `min_samples_leaf=9`,
  `n_estimators=50`), but the model is **17× larger**. It exceeds GitHub's file limit and the memory of the
  free hosting tier, while improving predictions by only ~20 seconds on average, so the smaller model stays in production.

## Approach

- **Data:** NYC TLC Yellow Taxi trips, January 2025 (3.47M trips, 3.21M after cleaning).
- **Cleaning:** trips inside January, distance 0–100 miles, duration 1–60 minutes, non-negative fare, speed < 80 mph.
- **Features:** pickup zone, dropoff zone, trip distance, pickup hour, pickup weekday, i.e. only information
  that is known *before* the trip. The same feature function is used in training and in the API.
- **Model:** `RandomForestRegressor` (`n_estimators=50`, `max_depth=10`, `min_samples_leaf=10`, `max_samples=0.1`),
  sized to train on an 8 GB laptop and to stay small enough for free hosting.
- **Tracking:** every training run and every Optuna trial (as nested runs) is logged to MLflow.

## Structure

```
taxi-duration-deployment/
├── src/
│   ├── data.py           # download and load the parquet data
│   ├── preprocessing.py  # cleaning, feature building, train/test split
│   ├── train.py          # Random Forest + baseline, logged to MLflow
│   ├── tune.py           # Optuna study, trials as nested MLflow runs
│   └── model.py          # load the model from a folder or from MLflow (MODEL_URI)
├── app/
│   └── main.py           # FastAPI app: GET / (health), POST /predict
├── artifacts/model/      # exported model used by the API and the Docker image
├── sample-request.json
├── Dockerfile
└── pyproject.toml
```

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
```

**API with the shipped model** (no MLflow server needed):

```bash
uv run uvicorn app.main:app --port 8000
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample-request.json
```

**Docker:**

```bash
docker build -t taxi-api .
docker run -p 8080:8080 taxi-api
```

**Training and tuning** (MLflow UI at http://localhost:5001):

```bash
uv run mlflow server --port 5001        # terminal 1 (port 5000 is taken by AirPlay on macOS)
uv run python -m src.train              # terminal 2
uv run python -m src.tune               # optional: Optuna study
```

To serve a model from MLflow instead of the shipped one, set `MODEL_URI=models:/<model-id>`.

## Challenges

- **Limited hardware:** 3.2M rows on an 8 GB MacBook Air; solved with shallow trees and per-tree subsampling.
- **Training vs. serving:** the API only knows pickup information, so feature building was separated from data cleaning.
- **Local setup:** MLflow returned `403` because macOS AirPlay occupies port 5000.
- **Free deployment:** the API cannot reach a local MLflow server, so the model is shipped inside the Docker image.

## Next steps

- Size-aware tuning (optimise RMSE and model size together) and more training data.
- Better features: zone pairs, boroughs, rush-hour and holiday flags; zone IDs as categories.
- Gradient boosting, validation on later months, tests for preprocessing and the API.
- Slimmer image without MLflow and a model registry instead of shipping the model in Git.
