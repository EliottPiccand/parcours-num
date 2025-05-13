import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def tune_knn_hyperparameters(data: pd.DataFrame):
    y = data['alerte']
    X = data[['timestamp', 'Valeur']].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=random_state)

    y_train = y_train.replace(0, -1)
    y_test = y_test.replace(0, -1)

    # Define a broader range of parameters to test
    param_grid = {
        'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], # Common metrics
        'p': [1, 2, 3] # Relevant for minkowski metric
    }

    knn = KNeighborsClassifier()

    # Use GridSearchCV to find the best parameters
    # cv=5 means 5-fold cross-validation
    # scoring='accuracy' is the metric used to evaluate models
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1) # n_jobs=-1 uses all available cores
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_


def train_model_knn(data: pd.DataFrame):
    best_params = tune_knn_hyperparameters(data)

    y = data['alerte']
    X = data[['timestamp', 'Valeur']].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=random_state)

    y_train = y_train.replace(0, -1)
    y_test = y_test.replace(0, -1)

    # Train the model using the best parameters found
    knn = KNeighborsClassifier(**best_params)
    knn.fit(X_train, y_train)

    return knn, X_test, y_test


def get_model_knn_error(model: KNeighborsClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    return error_rate