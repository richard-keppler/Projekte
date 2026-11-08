# King County House Prices – Refactoring & CRUD API

Starting point was a working but monolithic Jupyter notebook analysing house sales in King County (Seattle area). I refactored its **data cleaning** and **feature engineering** into reusable, side-effect-free Python functions and pipelines, and built a **FastAPI CRUD service** that stores houses in **PostgreSQL**.

**Skills shown:** refactoring notebook code into modules · pandas · NumPy · geo-distance features · FastAPI · Pydantic validation · SQLAlchemy 2.0 ORM · PostgreSQL · Docker (database container) · REST API design

## Structure

```
house-price-refactoring/
├── cleaning.py             # cleaning steps + cleaning_pipeline()
├── feature_engineering.py  # feature steps + feature_engineering()
├── app/
│   ├── database.py         # engine, session handling (connection string via env var)
│   ├── models.py           # SQLAlchemy table "houses"
│   ├── schemas.py          # Pydantic schemas for input/output validation
│   └── main.py             # CRUD endpoints
└── pyproject.toml
```

## Data pipeline

**Cleaning** (`cleaning.py`)

- remove an implausible record (33 bedrooms)
- recalculate `sqft_basement` as `sqft_living − sqft_above`
- fill missing values in `view` and `waterfront`
- combine `yr_built` and `yr_renovated` into `last_known_change`

**Feature engineering** (`feature_engineering.py`)

- price per square foot
- distance to the wealth centre (Medina), using a latitude-corrected distance in km
- distance to the nearest waterfront property

Each step is a separate function that works on a copy of the DataFrame, so the input is never modified.

```python
from cleaning import load_data, cleaning_pipeline
from feature_engineering import feature_engineering

df = load_data("King_County_House_prices_dataset.csv")
df = feature_engineering(cleaning_pipeline(df))
```

The dataset is the public [King County House Sales dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) (21,597 sales, 2014–2015).

## CRUD API

Stores houses with five features (`bedrooms`, `bathrooms`, `sqft_living`, `grade`, `zipcode`).

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/houses` | create a house (201) |
| `GET` | `/houses` | list all houses |
| `GET` | `/houses/{id}` | get one house (404 if unknown) |
| `PUT` | `/houses/{id}` | update a house |
| `DELETE` | `/houses/{id}` | delete a house (204) |

Input is validated with Pydantic (e.g. `grade` between 1 and 13, positive living area); invalid requests return 422.

```bash
docker run -d --name houses-db -e POSTGRES_PASSWORD=<password> -p 5432:5432 postgres
export DB_CONN="postgresql+psycopg://postgres:<password>@localhost:5432/postgres"
uv sync
uv run uvicorn app.main:app --reload    # Swagger UI: http://127.0.0.1:8000/docs
```

## Lessons learned

- **No side effects:** every function starts with `data.copy()` and assigns results instead of using `inplace=True`.
- **Debugging with shapes:** a wrong indentation in a nested loop produced 3,153,016 values instead of 21,596 – the length mismatch revealed the bug.
- **Models vs. schemas:** SQLAlchemy models describe storage, Pydantic schemas describe what clients may send and receive (e.g. no `id` on input).

**Next steps:** vectorise the waterfront distance with NumPy (currently ~3 million distance calculations in a loop), and use fixed reference coordinates from the training data so the pipeline can be applied to new data.
