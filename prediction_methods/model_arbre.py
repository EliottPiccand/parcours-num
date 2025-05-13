from pandas import DataFrame

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

import scipy as sp

import numpy as np

# Select random seed
random_state = None 

type X = ...

def train_model_arbre_classification(data_train: DataFrame) -> DecisionTreeClassifier: 
    nbdepth = 4
    My_tree = DecisionTreeClassifier(max_depth=nbdepth,criterion='gini')
    x_train = np.array(data_train["timestamp"], data_train["Valeur"])
    y_train = data_train["alerte"]  # état d'alerte ou non : état d'alerte : 1, pas d'alerte : 0

    My_tree = My_tree.fit(x_train, y_train)    
    return My_tree


def train_model_arbre_regression(data_train: DataFrame) -> X:
    nbdepth = 4
    regr = DecisionTreeRegressor(max_depth=nbdepth, criterion='squared_error')
    x_train = data_train["timestamp"]
    y_train = data_train["Valeur"]  # état d'alerte ou non : état d'alerte : 1, pas d'alerte : 0

    regr = regr.fit(x_train, y_train)
    return regr


def get_model_error_arbre(model : DecisionTreeClassifier, data_test: DataFrame) -> float:
    error = 0
    

    return error

