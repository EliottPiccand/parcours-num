from os import getenv

from numpy import array
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from .format_dataset import format_dataset, has_alert_been_raised_next_day


def tune_knn_hyperparameters(x, y):
    if getenv("USE_REDUCED_DATA", "False") == "True":
        param_grid = {
            "n_neighbors": [3, 5],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
            "p": [1, 2],
        }
    else:
        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
            "p": [1, 2, 3],
        }

    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(x, y)

    return grid_search.best_params_

HOURS = [0, -24, -48]
INPUT_FEATURES = [
    f"timestamp_h{hour}"
    for hour in HOURS
] + [
    f"Valeur_h{hour}"
    for hour in HOURS
]
def train_model_knn_classification(data: DataFrame) -> KNeighborsClassifier:
    data = format_dataset(data, HOURS)

    x = data[INPUT_FEATURES].to_numpy()
    y = data["alerte_d+1"].to_numpy()

    print("   - Performing cross validation")
    best_params = tune_knn_hyperparameters(x, y)
    print("     > Best params :", best_params)

    print("   - Fitting best model found")
    knn = KNeighborsClassifier(**best_params)
    knn.fit(x, y)

    return knn

def get_model_knn_classification_error(model: KNeighborsClassifier, data: DataFrame) -> float:
    data = format_dataset(data, HOURS)
    
    x = data[INPUT_FEATURES]
    
    print("   - Computing real values")
    y = array([
        has_alert_been_raised_next_day(data, ts)
        for ts in x["timestamp_h0"]
    ])
    x = x.to_numpy()

    print("   - Predicting values")
    y_pred = model.predict(x)

    print("   - Computing accuracy")
    return 1 - accuracy_score(y, y_pred)
