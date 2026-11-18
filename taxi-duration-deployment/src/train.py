from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.dummy import DummyRegressor
import mlflow
from src.data import download_data, load_data, DATA_URL, FILE_PATH
from src.preprocessing import clean_data, split_data

NAME_EXPERIMENT = "taxi-duration"
PARAMS = {"n_estimators": 50, "max_depth": 10, "min_samples_leaf": 10, "max_samples": 0.1, "n_jobs": -1, "random_state": 42}
TRACKING_URI = "http://localhost:5001"


def train_mlflow(parameters, x_train, y_train, x_test, y_test):
    with mlflow.start_run():
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
        model_info = mlflow.sklearn.log_model(sk_model=random_forest, name="taxi_duration", skops_trusted_types=["sklearn.tree._tree.Tree"], input_example=x_train.head(5))
        print(f"Training successful! \nRMSE: {rmse:.2f} \nDummy: {rmse_dummy:.2f}")


def main(tracking_uri, name_experiment, data_url, file_path):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(name_experiment)
    file_path = download_data(data_url, file_path)
    df = load_data(file_path)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    train_mlflow(PARAMS, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main(TRACKING_URI, NAME_EXPERIMENT, DATA_URL, FILE_PATH)