# Machine Learning, Deep Learning, NLP & ML Engineering – Projects

This repository collects projects from my M.Sc. studies in *Applied Artificial Intelligence* and my Machine Learning Engineering training. They cover medical image and text classification, time-series forecasting, data pipelines and putting ML code into production-ready shape.

**Technologies:** Python · TensorFlow / Keras · PyTorch · Hugging Face Transformers · scikit-learn · XGBoost · Optuna · SHAP · Pandas / NumPy · SQL / DuckDB · Prefect · MLflow · FastAPI · SQLAlchemy · PostgreSQL · Docker · Render · pytest · uv

---

## Machine Learning & Deep Learning

### 1. Pneumonia Detection
Binary classification of chest X-rays (*normal* vs. *pneumonia*).

- **Approach:** custom CNN with four convolutional blocks, dropout and early stopping.
- **Data pipeline:** pixel normalisation and data augmentation (flip, rotation).
- **Technologies:** TensorFlow / Keras, NumPy, Matplotlib.
- 📄 [`ML Pneumonia Detection.py`](ML%20Pneumonia%20Detection.py)

### 2. Lung Disease Classification
Multi-class classification of X-ray images into **four disease classes**.

- **Approach:** CNN-LSTM hybrid model with systematic hyperparameter tuning.
- **Class imbalance:** class weights and data augmentation (Albumentations).
- **Result:** about **89 % test accuracy**.
- **Technologies:** TensorFlow / Keras, Albumentations, OpenCV, NumPy, SHAP.
- 📓 [`ML Lung Disease Classification.ipynb`](ML%20Lung%20Disease%20Classification.ipynb)

### 3. Disease Categorisation (NLP)
NLP pipeline that assigns unstructured medical texts to **five disease categories**.

- **Approach:** domain-specific language model (**Bio_ClinicalBERT**) for fine-tuning and feature extraction, combined with a stacking ensemble (XGBoost / logistic regression).
- **Class imbalance:** SMOTE and class weights; hyperparameter optimisation with Optuna, model interpretation with SHAP.
- **Technologies:** PyTorch, Hugging Face Transformers, scikit-learn, XGBoost, Optuna, SHAP.
- 📓 [`NLP Disease Categorisation.ipynb`](NLP%20Disease%20Categorisation.ipynb)

### 4. Electricity Price Forecasting
Forecasting hourly electricity prices from historical energy and weather data.

- **Data preparation:** merging energy and weather time series, ARIMA-based imputation of missing values, sequence generation and min-max scaling.
- **Approach:** comparison of two deep learning architectures (**LSTM vs. GRU**) with early stopping.
- **Result:** the LSTM produced more accurate forecasts (R² ≈ 0.44, MAE ≈ 5.2) than the GRU (R² ≈ 0.16, MAE ≈ 6.5).
- **Technologies:** PyTorch, statsmodels (ARIMA), scikit-learn, Pandas, Matplotlib.
- 📓 [`DL Electricity Price Forecasting.ipynb`](DL%20Electricity%20Price%20Forecasting.ipynb)

---

## Data & ML Engineering

### 5. NYC Taxi Revenue Pipeline (ELT)
Local ELT pipeline that downloads three months of NYC Green Taxi trip data (≈ 146,000 trips), loads it unchanged into DuckDB and calculates the daily revenue with SQL.

- **Approach:** separate extract, load and transform modules, a pipeline entry point and the same steps orchestrated as a **Prefect** flow.
- **Data quality:** documented cleaning rules based on the official data dictionary (refunds, disputes, out-of-period trips); idempotent steps.
- **Testing:** pytest tests with a synthetic DuckDB table covering every filter rule and idempotency.
- **Technologies:** Python, SQL, DuckDB, Prefect, pytest, uv.
- 📁 [`data-pipeline/`](data-pipeline/)

### 6. House Price Data – Refactoring & CRUD API
Refactoring of a monolithic data science notebook (King County house sales) into reusable code, plus a REST service.

- **Refactoring:** cleaning and feature engineering (incl. geo-distance features) extracted into side-effect-free functions and pipelines.
- **API:** FastAPI CRUD service with Pydantic validation, SQLAlchemy 2.0 ORM and PostgreSQL running in Docker.
- **Technologies:** Pandas, NumPy, FastAPI, Pydantic, SQLAlchemy, PostgreSQL, Docker.
- 📁 [`house-price-refactoring/`](house-price-refactoring/)

### 7. NYC Taxi Trip Duration – Training, Tracking & Deployment
End-to-end ML project that predicts the duration of NYC Yellow Taxi trips (January 2025, ≈ 3.2M trips) and serves the model as a public API.

- **Approach:** Random Forest with a mean baseline, experiment tracking with MLflow and hyperparameter tuning with Optuna (trials as nested MLflow runs).
- **Serving:** FastAPI `/predict` endpoint with Pydantic validation; the same feature function is used in training and serving.
- **Deployment:** Docker image deployed on Render, with a [live demo](https://taxi-duration-api.onrender.com/docs).
- **Result:** test RMSE of **4.39 minutes** (baseline 9.77, −55 %). Optuna reached 4.06, but the model was 17× larger, so the smaller model was kept in production.
- **Technologies:** scikit-learn, MLflow, Optuna, FastAPI, Pydantic, Docker, Render, uv.
- 📁 [`taxi-duration-deployment/`](taxi-duration-deployment/)

---

*The projects are self-contained case studies that deepen the practical side of my studies and training.*
