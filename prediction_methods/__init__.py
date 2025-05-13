from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame

    type Model = callable[[DataFrame], bool]
    type TrainModel = callable[[DataFrame], Model]
    type GetModelError = callable[[Model, DataFrame], float]


PREDICTION_METHODS: tuple[tuple[str, TrainModel, GetModelError], ...] = (
    ("name", ..., ...),
)
