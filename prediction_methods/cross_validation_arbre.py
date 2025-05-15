from pandas import DataFrame

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

import scipy as sp

import numpy as np


def cross_validation_arbre_classification(x, y) -> int:
    param_grid = {
        "depth_array" : np.arange(1, 20)
        "metric" : ['gini', 'entropy']
    }
    My_tree_class = DecisionTreeClassifier()
    grid_search = GridSearchCV(My_tree_class, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(x, y)

    return grid_search.best_params_



def cross_validation_arbre_regression(x, y) -> int:

    param_grid = {
        "depth_array" : np.arange(1, 20)
        "metric" : ['squared_error', 'friedman_mse']
    }
    My_tree_class = DecisionTreeRegressor()
    grid_search = GridSearchCV(My_tree_class, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(x, y)

    return grid_search.best_params_

    