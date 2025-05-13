from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import unique

from .remove_missing_values import remove_missing_values

if TYPE_CHECKING:
    from pandas import DataFrame


def fill_with_next(data: DataFrame) -> DataFrame:
    """Fill nan values with next non-nan value, per station
    
    If station data ends with nan, fill with last non-nan value
    Final removal of nan still present because some stations have only nan
    """
    data = data.copy()
    for station_id in unique(data["idPolair"]):
        data[(data["idPolair"] == station_id)] = data[(data["idPolair"] == station_id)].bfill().ffill()
    return remove_missing_values(data)

