from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from .format_dataset import format_dataset, has_alert_been_raised_next_day

def tune_random_forest_hyperparameters_regression(x, y):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

    grid_search.fit(x, y.ravel())

    return grid_search.best_params_

def train_model_random_forest_regression(data: DataFrame) -> RandomForestRegressor:
    data_formatted = format_dataset(data, [0, -24, -48])

    if 'regression_target' not in data_formatted.columns:
         raise KeyError("Target column 'regression_target' not found. Please update the code with your actual target column name.")
    y = data_formatted["regression_target"]

    x = data_formatted[["timestamp_h0", "Valeur_h0", "timestamp_h-24", "Valeur_h-24", "timestamp_h-48", "Valeur_h-48"]]

    print("   - Performing cross validation")
    best_params = tune_random_forest_hyperparameters_regression(x, y)

    print("   - Fitting best model found")
    rf = RandomForestRegressor(**best_params, random_state=42)
    rf.fit(x, y.ravel())

    return rf

def get_model_random_forest_error_regression(model: RandomForestRegressor, data: DataFrame) -> float:
    data = format_dataset(data, [0, -24, -48])

    x = data[["timestamp_h0", "Valeur_h0", "timestamp_h-24", "Valeur_h-24", "timestamp_h-48", "Valeur_h-48"]]

    print("   - Computing real values")
    y = [
        has_alert_been_raised_next_day(data, ts)
        for ts in x["timestamp_h0"]
    ]

    print("   - Predicting values")
    y_pred = model.predict(x)

    print("   - Computing accuracy")
    error = 1 - accuracy_score(y, y_pred)
    return error