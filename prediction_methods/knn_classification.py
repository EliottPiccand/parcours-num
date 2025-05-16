from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from .format_dataset import format_dataset, has_alert_been_raised_next_day


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

def train_model_knn_classification(data: DataFrame) -> KNeighborsClassifier:
    data = format_dataset(data, [0, -24, -48])
    y = data["alerte_d+1"]
    x = data[["timestamp_h0", "Valeur_h0", "timestamp_h-24", "Valeur_h-24", "timestamp_h-48", "Valeur_h-48"]]

    print("   - Performing cross validation")
    best_params = tune_knn_hyperparameters(x, y)

    print("   - Fitting best model found")
    knn = KNeighborsClassifier(**best_params)
    knn.fit(x, y)

    return knn

def get_model_knn_classification_error(model: KNeighborsClassifier, data: DataFrame) -> float:
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
    accuracy = accuracy_score(y, y_pred)
    error_rate = 1 - accuracy
    return error_rate