import pandas as DataFrame
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
from .format_dataset import format_dataset, has_alert_been_raised_next_day

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

def tune_knn_hyperparameters(x, y):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
        'p': [1, 2, 3]
    }

    knn = KNeighborsRegressor()

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=mse_scorer, n_jobs=-1)
    grid_search.fit(x, y)

    return grid_search.best_params_


def train_model_knn(data: DataFrame) -> KNeighborsRegressor:
    
    data = format_dataset(data, [0, -24, -48])  
    y = data['alerte_d+1']   
    x = data[['timestamp_h0', 'Valeur_h0', 'timestamp_h-24', 'Valeur_h-24', 'timestamp_h-48', 'Valeur_h-48']]

    best_params = tune_knn_hyperparameters(x, y)

    knn = KNeighborsRegressor(**best_params)
    knn.fit(x, y)

    return knn


def get_model_knn_error(model: KNeighborsRegressor, data: DataFrame) -> float:
  
    data = format_dataset(data, [0, -24, -48])

    x = data[['timestamp_h0', 'Valeur_h0', 'timestamp_h-24', 'Valeur_h-24', 'timestamp_h-48', 'Valeur_h-48']].copy()

    y = []
    for ts in x['timestamp_h0']:
        y.append(has_alert_been_raised_next_day(data, ts))

    y_pred = model.predict(x)
    
    error = 1 - accuracy_score(y, y_pred)
    
    return error