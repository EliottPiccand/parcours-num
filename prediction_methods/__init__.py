from __future__ import annotations

from typing import TYPE_CHECKING

from .knn_classification import get_model_knn_error, train_model_knn
from .decision_tree_classification import (
    get_model_decision_tree_classification_error,
    train_model_decision_tree_classification,
)
from .decision_tree_regression import (
    get_model_decision_tree_regression_error,
    train_model_decision_tree_regression,
)

if TYPE_CHECKING:
    from typing import Any

    from pandas import DataFrame

    type Model = Any
    type TrainModel = callable[[DataFrame], Model]
    type GetModelError = callable[[Model, DataFrame], float]


PREDICTION_METHODS: tuple[tuple[str, TrainModel, GetModelError], ...] = (
    # ("knn classifier", train_model_knn, get_model_knn_error),
    ("tree classifier", train_model_decision_tree_classification, get_model_decision_tree_classification_error),
    ("tree regressor", train_model_decision_tree_regression, get_model_decision_tree_regression_error),
)
