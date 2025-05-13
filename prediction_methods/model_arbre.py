from pandas import DataFrame

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

import scipy as sp

import numpy as np

from .format_dataset import format_dataset, has_alert_been_raised_next_day

# Select random seed
random_state = None 

type X = ...

def train_model_arbre_classification(data_train: DataFrame) -> DecisionTreeClassifier: 
    data = format_dataset(sata_train, [0, -24, -48])

    x_train = data_train[['timestamp_h0', 'Valeur_h0', 'timestamp_h-24', 'Valeur_h-24', 'timestamp_h-48', 'Valeur_h-48']]
    y_train = data_train["alerte"]  # état d'alerte ou non : état d'alerte : 1, pas d'alerte : 0
    best_params = cross_validation_arbre_classification(x, y)

    My_tree = DecisionTreeClassifier(**best_params)

    My_tree = My_tree.fit(x_train, y_train)    
    return My_tree

def get_model_error_arbre_classification(model : DecisionTreeClassifier, data_test: DataFrame) -> float:
    x = format_dataset(data_test, [0, -24, -48])[['timestamp_h0', 'Valeur_h0', 'timestamp_h-24', 'Valeur_h-24', 'timestamp_h-48', 'Valeur_h-48']]
    
    y = []
    for temps in x["timestamp_h0"]:
        y.append(has_alert_been_raised_next_day(data_test, temps))

    y_prediction = model.predict(x)
    error = 1 - accuracy_score(y, y_prediction)
    return error


def train_model_arbre_regression(data_train: DataFrame) -> X:
    data = format_dataset(sata_train, [0, -24, -48])

    x = data_train[["timestamp_h0", "timestamp_h-24", "timestamp_h-48"]]
    y = data_train["Valeur"]  # état d'alerte ou non : état d'alerte : 1, pas d'alerte : 0
    best_params = cross_validation_arbre_regression(x, y)

    regr = DecisionTreeRegressor(**best_params)

    regr = regr.fit(x, y)
    return regr



def get_model_error_arbre_regression(model : DecisionTreeClassifier, data_test: DataFrame) -> float:
    x = format_dataset(data_test, [0, -24, -48])[["timestamp_h0", "timestamp_h-24", "timestamp_h-48"]]
    
    y = []
    for temps in x["timestamp_h0"]:
        y.append(has_alert_been_raised_next_day(data_test, temps))

    y_prediction = model.predict(x)
    error = 1 - accuracy_score(y, y_prediction)
    return error