# Machine Learning, Deep Learning, NLP & ML Engineering – Projects

This repository collects projects from my M.Sc. studies in *Applied Artificial Intelligence* and my Machine Learning Engineering training. They cover medical image and text classification, time-series forecasting, data pipelines and putting ML code into production-ready shape.

**Technologies:** Python · TensorFlow / Keras · PyTorch · Hugging Face Transformers · scikit-learn · XGBoost · Optuna · SHAP · Pandas / NumPy · SQL / DuckDB · Prefect · FastAPI · SQLAlchemy · PostgreSQL · Docker · pytest · uv

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
- 📓 [`ML Hyperparameter Tuning.ipynb`](ML%20Hyperparameter%20Tuning.ipynb)

### 3. Disease Categorisation (NLP)
NLP pipeline that assigns unstructured medical texts to **five disease categories**.

- **Approach:** domain-specific language model (**Bio_ClinicalBERT**) for fine-tuning and feature extraction, combined with a stacking ensemble (XGBoost / logistic regression).
- **Class imbalance:** SMOTE and class weights; hyperparameter optimisation with Optuna, model interpretation with SHAP.
- **Technologies:** PyTorch, Hugging Face Transformers, scikit-learn, XGBoost, Optuna, SHAP.
- 📓 [`NLP Kategorisierung Krankheit.ipynb`](NLP%20Kategorisierung%20Krankheit.ipynb)

### 4. Electricity Price Forecasting
Forecasting hourly electricity prices from historical energy and weather data.

- **Data preparation:** merging energy and weather time series, ARIMA-based imputation of missing values, sequence generation and min-max scaling.
- **Approach:** comparison of two deep learning architectures (**LSTM vs. GRU**) with early stopping.
- **Result:** the LSTM produced more accurate forecasts (R² ≈ 0.44, MAE ≈ 5.2) than the GRU (R² ≈ 0.16, MAE ≈ 6.5).
- **Technologies:** PyTorch, statsmodels (ARIMA), scikit-learn, Pandas, Matplotlib.
- 📓 [`DL Vorhersage von Strompreisen.ipynb`](DL%20Vorhersage%20von%20Strompreisen.ipynb)

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

---

*The projects are self-contained case studies that deepen the practical side of my studies and training.*
