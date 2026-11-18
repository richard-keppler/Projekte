import mlflow
import optuna
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from src.data import DATA_URL, FILE_PATH, download_data, load_data
from src.preprocessing import clean_data, split_data

NAME_EXPERIMENT = "taxi-duration"
TRACKING_URI = "http://localhost:5001"
N_TRIALS = 20
FIXED_PARAMS = {"max_samples": 0.1, "n_jobs": -1, "random_state": 42}


def objective(trial, x_train, y_train, x_test, y_test):
    tuned_params = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 100, step=10),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
    }
    parameters = {**FIXED_PARAMS, **tuned_params}
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        random_forest = RandomForestRegressor(**parameters)
        random_forest.fit(x_train, y_train)
        y_pred = random_forest.predict(x_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mlflow.log_params(parameters)
        mlflow.log_metric("RMSE", rmse)
    return rmse


def train_mlflow(parameters, x_train, y_train, x_test, y_test):
    with mlflow.start_run(run_name="best-model", nested=True):
        mlflow.log_params(parameters)
        random_forest = RandomForestRegressor(**parameters)
        random_forest.fit(x_train, y_train)
        y_pred = random_forest.predict(x_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mlflow.log_metric("RMSE", rmse)
        dummy_regressor = DummyRegressor()
        dummy_regressor.fit(x_train, y_train)
        y_pred_dummy = dummy_regressor.predict(x_test)
        rmse_dummy = root_mean_squared_error(y_test, y_pred_dummy)
        mlflow.log_metric("RMSE_baseline", rmse_dummy)
        model_info = mlflow.sklearn.log_model(
            sk_model=random_forest,
            name="taxi_duration",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            input_example=x_train.head(5),
        )
        print(f"Best model RMSE: {rmse:.2f} | Baseline: {rmse_dummy:.2f}")
        print(f"Model URI: {model_info.model_uri}")


def main(tracking_uri, name_experiment, data_url, file_path):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(name_experiment)
    file_path = download_data(data_url, file_path)
    df = load_data(file_path)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    with mlflow.start_run(run_name="optuna-study"):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_test, y_test),
            n_trials=N_TRIALS,)
        print(f"Best RMSE: {study.best_value:.2f}")
        print(f"Best params: {study.best_params}")
        mlflow.log_metric("best_RMSE", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        best_params = {**FIXED_PARAMS, **study.best_params}
        train_mlflow(best_params, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main(TRACKING_URI, NAME_EXPERIMENT, DATA_URL, FILE_PATH)