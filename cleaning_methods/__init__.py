from __future__ import annotations

from typing import TYPE_CHECKING

from .fill_with_next import fill_with_next
from .fill_with_previous import fill_with_previous
from .remove_missing_values import remove_missing_values
from .guesstimate_sin import guesstimate_sin

if TYPE_CHECKING:
    from pandas import DataFrame

    type CleaningMethod = callable[[DataFrame], DataFrame]


CLEANING_METHODS: tuple[tuple[str, CleaningMethod], ...] = (
    ("guesstimate_sin", guesstimate_sin),
    ("remove missing values", remove_missing_values),
    ("fill with previous", fill_with_previous),
    ("fill with next", fill_with_next),
)
