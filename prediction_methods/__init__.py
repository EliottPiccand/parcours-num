from __future__ import annotations

from typing import TYPE_CHECKING
# from .model_arbre import train_model_arbre_classification, get_model_error_arbre
# from .model_arbre import train_model_arbre_regression
from .knn import train_model_knn, get_model_knn_error
from .model_arbre import train_model_arbre_classification, get_model_error_arbre_classification
from .model_arbre import train_model_arbre_regression, get_model_error_arbre_regression

if TYPE_CHECKING:
    from typing import Any

    from pandas import DataFrame

    type Model = Any
    type TrainModel = callable[[DataFrame], Model]
    type GetModelError = callable[[Model, DataFrame], float]


PREDICTION_METHODS: tuple[tuple[str, TrainModel, GetModelError], ...] = (
    #("knn classifier", train_model_knn, get_model_knn_error),
    ("tree classifier", train_model_arbre_classification, get_model_error_arbre_classification),
    ("tree regressor", train_model_arbre_regression, get_model_error_arbre_regression),
)
