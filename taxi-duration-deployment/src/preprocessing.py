from sklearn.model_selection import train_test_split

FEATURES = ["PULocationID", "DOLocationID", "trip_distance", "pickup_hour", "pickup_dayofweek"]
TARGET = "duration"


def convert_hour_weekday(df):
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.day_of_week
    return df


def clean_data(dataframe):
    drop_columns = ["VendorID", "passenger_count", "store_and_fwd_flag"]
    df = dataframe.drop(columns=drop_columns)
    df = df[(df["tpep_pickup_datetime"] >= "2025-01-01") & (df["tpep_pickup_datetime"] < "2025-02-01")]
    df = df[(df["tpep_dropoff_datetime"] >= "2025-01-01") & (df["tpep_dropoff_datetime"] < "2025-02-01")]
    df["duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
    df = df[df["duration"].between(1, 60)]
    df = df[df["fare_amount"] >= 0]
    df = df[df["trip_distance"] / (df["duration"] / 60) < 80]
    df = convert_hour_weekday(df)
    df = df[FEATURES + [TARGET]]
    return df


def split_data(dataframe):
    X = dataframe[FEATURES]
    y = dataframe[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    return X_train, X_test, y_train, y_test
