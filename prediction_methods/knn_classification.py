from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from .format_dataset import format_dataset, has_alert_been_raised_next_day

if TYPE_CHECKING:
    from pandas import DataFrame


def tune_knn_hyperparameters(x, y):
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
        "p": [1, 2, 3]
    }

    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(x, y)

    return grid_search.best_params_

def train_model_knn(data: DataFrame) -> KNeighborsClassifier:
    data = format_dataset(data, [0, -24, -48])
    y = data["alerte_d+1"]
    x = data[["timestamp_h0", "Valeur_h0", "timestamp_h-24", "Valeur_h-24", "timestamp_h-48", "Valeur_h-48"]]

    best_params = tune_knn_hyperparameters(x, y)

    knn = KNeighborsClassifier(**best_params)
    knn.fit(x, y)

    return knn

def get_model_knn_error(model: KNeighborsClassifier, data: DataFrame) -> float:
    formatted_data = format_dataset(data, [0, -24, -48])
    x = formatted_data[["timestamp_h0", "Valeur_h0", "timestamp_h-24", "Valeur_h-24", "timestamp_h-48", "Valeur_h-48"]]
    
    y = [
        has_alert_been_raised_next_day(data, ts)
        for ts in x["timestamp_h0"]
    ]

    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    error_rate = 1 - accuracy
    return error_rate
