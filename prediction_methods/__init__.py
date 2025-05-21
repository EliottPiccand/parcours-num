from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import arange
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .predictor import Predictor

if TYPE_CHECKING:
    from typing import Any

    from pandas import DataFrame

    type Model = Any
    type TrainModel = callable[[DataFrame], Model]
    type GetModelError = callable[[Model, DataFrame], float]


__all__ = (
    "PREDICTION_METHODS",
    "Predictor",
)

PREDICTION_METHODS: tuple[tuple[Model, bool, list[int], dict, dict], ...] = (
    # (
    #     KNeighborsClassifier,
    #     False,
    #     [0, -24, -48],
    #     {
    #         "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    #         "weights": ["uniform", "distance"],
    #         "metric": ["chebyshev", "minkowski"],
    #         "p": [1, 2, 3],
    #     },
    #     {
    #         "n_neighbors": [3, 5],
    #         "weights": ["uniform", "distance"],
    #         "metric": ["chebyshev", "minkowski"],
    #         "p": [1, 2],
    #     },
    # ),
    # (
    #     KNeighborsRegressor,
    #     True,
    #     [0, -24, -48],
    #     {
    #         "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    #         "weights": ["uniform", "distance"],
    #         "metric": ["chebyshev", "minkowski"],
    #         "p": [1, 2, 3],
    #     },
    #     {
    #         "n_neighbors": [3, 5],
    #         "weights": ["uniform", "distance"],
    #         "metric": ["chebyshev", "minkowski"],
    #         "p": [1, 2],
    #     },
    # ),
    (
        RandomForestClassifier,
        False,
        [0, -24, -48],
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        },
        {
            "n_estimators": [50, 60],
            "max_depth": [4, 6],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "criterion": ["gini", "entropy"]
        },
    ),
    (
        RandomForestRegressor,
        True,
        [0, -24, -48],
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["absolute_error", "squared_error", "poisson", "friedman_mse"]
        },
        {
            "n_estimators": [50, 60],
            "max_depth": [4, 6],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "criterion": ["absolute_error", "squared_error"]
        },
    ),
    (
        DecisionTreeClassifier,
        False,
        [0, -24, -48],
        {
            "max_depth" : arange(1, 20),
            "criterion" : ["gini", "entropy", "log_loss"],
        },
        {
            "max_depth" : [1, 2],
            "criterion" : ["gini", "entropy"],
        },
    ),
    (
        DecisionTreeRegressor,
        True,
        [0, -24, -48],
        {
            "max_depth" : arange(1, 20),
            "criterion" : ["friedman_mse", "squared_error", "poisson", "absolute_error"],
            "splitter"  : ["best", "random"],
        },
        {
            "max_depth" : [1, 2],
            "criterion" : ["friedman_mse", "squared_error"],
            "splitter"  : ["best", "random"],
        },
    ),
)
