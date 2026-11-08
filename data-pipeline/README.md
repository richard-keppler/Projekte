# NYC Green Taxi – Daily Revenue Pipeline (ELT)

A local **ELT pipeline** that downloads the NYC Green Taxi trip records for January–March 2025, loads them unchanged into a **DuckDB** database and calculates the **revenue per day** with SQL inside the database. The same steps are also orchestrated as a **Prefect** flow.

**Skills shown:** ELT design · SQL (DuckDB) · Python packaging with `uv` · workflow orchestration (Prefect) · unit testing with pytest · idempotent pipelines · data-quality rules based on the source's data dictionary

```
Extract                     Load                            Transform
NYC TLC website  ──▶  data/raw/*.parquet  ──▶  DuckDB: green_taxi  ──▶  daily_revenue (table + CSV)
(download)            (staging, never edited)  (raw, unchanged)         SQL aggregation
```

## Structure

```
data-pipeline/
├── src/taxi_pipeline/
│   ├── extract.py      # download the monthly Parquet files; defines all paths
│   ├── load.py         # raw files -> DuckDB table green_taxi
│   ├── transform.py    # green_taxi -> daily_revenue (SQL)
│   ├── pipeline.py     # runs extract, load, transform in order
│   └── flow.py         # the same steps as Prefect tasks in one flow
├── tests/
│   ├── conftest.py     # puts src/taxi_pipeline on the import path
│   └── test_transform.py
└── pyproject.toml
```

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13. Run all commands from this folder.

```bash
uv sync
uv run python src/taxi_pipeline/pipeline.py   # whole pipeline
uv run python src/taxi_pipeline/flow.py       # same pipeline as a Prefect flow
uv run pytest                                 # tests
```

Optional: `uv run prefect server start` shows all flow and task runs in the Prefect UI.

Expected output:

```
Loading successful! 146486 rows loaded into green_taxi.
Transform successful! 90 days, 144895 trips -> .../data/processed/daily_revenue.csv
```

## Design decisions

**Why ELT instead of ETL:** The raw data is loaded unchanged (`SELECT *`) and all cleaning happens afterwards in SQL. If the definition of revenue or a filter rule changes, only the transform step has to be re-run – no new download or load. DuckDB reads Parquet natively, is built for analytical queries and needs no server.

**Definition of revenue:** `total_amount` ("the total amount charged to passengers" according to the TLC data dictionary). It already contains credit card tips, so `tip_amount` is not added on top.

**Cleaning rules:**

| Rule | Reason |
|---|---|
| A trip belongs to the day of `lpep_pickup_datetime` | moment the meter was engaged; 1,241 trips cross midnight |
| Only pickups from 2025-01-01 to 2025-03-31 | 11 trips lie outside the requested period |
| Only `total_amount > 0` | negative amounts (refunds) and zero amounts are not revenue |
| Exclude `payment_type` 3 (no charge) and 4 (dispute) | no money was collected, or it is uncertain |
| Keep trips with missing `payment_type` | amounts are normal; only the payment info is missing |

In total **1,591 of 146,486 trips (≈ 1.1 %)** are excluded.

**Idempotency:** Every step can be re-run with the same result – downloads overwrite files, load and transform use `CREATE OR REPLACE TABLE`. A test checks this explicitly.

## Lessons learned

- Building paths from `__file__` instead of relative paths, so scripts work from any working directory.
- DuckDB specifics: persistent vs. in-memory connections, `COPY … TO` for exports, parameters only for values (not table names), single-writer lock.
- Ambiguous requirements ("revenue") need a documented decision based on the data dictionary.

**Next steps:** make the period configurable, pass target folders as parameters and add tests for extract and load.
