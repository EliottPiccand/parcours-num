from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame


def remove_missing_values(data: DataFrame) -> DataFrame:
    """Remove all rows with a nan value"""
    return data[(data["Valeur"].notna())]
