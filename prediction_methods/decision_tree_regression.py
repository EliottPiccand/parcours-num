from numpy import arange
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from .format_dataset import format_dataset, has_alert_been_raised_next_day


def tune_best_parameters(x, y):
    param_grid = {
        "max_depth" : arange(1, 20),
        "criterion" : ["squared_error", "friedman_mse"],
    }
    tree = DecisionTreeRegressor()

    grid_search = GridSearchCV(tree, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(x, y)

    return grid_search.best_params_

def train_model_decision_tree_regression(data_train: DataFrame) -> DecisionTreeRegressor:
    data = format_dataset(data_train, [0, -24, -48])

    x = data[["timestamp_h0", "timestamp_h-24", "timestamp_h-48"]]
    y = data["Valeur_d+1"]

    print("   - Performing cross validation")
    best_params = tune_best_parameters(x, y)
    print("     > Best params :", best_params)

    print("   - Fitting best model found")
    tree = DecisionTreeRegressor(**best_params)
    tree.fit(x, y)

    return tree

def get_model_decision_tree_regression_error(model: DecisionTreeRegressor, data_test: DataFrame) -> float:
    x = format_dataset(data_test, [0, -24, -48])[["timestamp_h0", "timestamp_h-24", "timestamp_h-48"]]
    
    print("   - Computing real values")
    y = [
        has_alert_been_raised_next_day(data_test, timestamp)
        for timestamp in x["timestamp_h0"]
    ]

    print("   - Predicting values")
    y_pred = model.predict(x)
    for i in range(len(y_pred)):
        if y_pred[i] > 100:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    print("   - Computing accuracy")
    return 1 - accuracy_score(y, y_pred)
