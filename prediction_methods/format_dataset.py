from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame


def format_dataset(data: DataFrame, time_span: int, forecast: int) -> DataFrame:
    raise NotImplementedError