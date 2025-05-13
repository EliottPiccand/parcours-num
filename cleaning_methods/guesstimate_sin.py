from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from numpy import clip, inf, pi, sin, unique, where
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from pandas import DataFrame


W_ONE_YEAR = 2 * pi / (365.25 * 24 * 3600)
W_TWO_WEEKS = 2 * pi / (7 * 24 * 3600)
W_ONE_DAY = 2 * pi / (24 * 3600)


def model_station(t, a, phi_a, b, phi_b, c, phi_c, w, phi_d, offset):
    return clip(
        a * sin(W_ONE_YEAR * t - phi_a) + b * sin(W_TWO_WEEKS * t - phi_b) + c * sin(W_ONE_YEAR/w * t - phi_d) * sin(W_ONE_DAY * t - phi_c) + offset,
        0, inf
    )

STATION_KEEP_THRESHOLD = 4 # months
def get_station_id_with_enough_data(data: DataFrame) -> list[int]:
    """Keep only stations with at least STATION_KEEP_THRESHOLD months of data"""
    id_stations_to_keep = []
    for id_station in unique(data["idPolair"]):
        values = data[data["idPolair"] == id_station]
        nan_indices = where(values["Valeur"].isna())[0]

        nan_ratio = len(nan_indices) / len(values)

        if nan_ratio < (12 - STATION_KEEP_THRESHOLD) / 12:
            id_stations_to_keep.append(id_station)
    
    return id_stations_to_keep

YEAR_SLICE_COUNT = 6
WINDOW_OVERLAP = 1
def guesstimate_sin(data: DataFrame) -> DataFrame:
    """Guesstimate the missing values using a sum of sinus of periods :
        - one year
        - two weeks
        - one day

    with the 'one day' term modulated by another sinus with a period of one year

    Fitting the model using scipy.curve_fit (least squares method).
    For each station :
        - slice the year into YEAR_SLICE_COUNT parts
        - train the model on the the part + WINDOW_OVERLAP months on each side
    """
    MAX_TIME = max(data["timestamp"])
    MIN_TIME = min(data["timestamp"])
    for s, station_id in enumerate(get_station_id_with_enough_data(data)):
        station_values = data[data["idPolair"] == station_id]
        nan_indices = where(station_values["Valeur"].isna())[0]
        not_nan_indices = where(station_values["Valeur"].notna())[0]
        nan_values = station_values.iloc[nan_indices]
        not_nan_values = station_values.iloc[not_nan_indices]

        for i in range(YEAR_SLICE_COUNT):
            print(s, i)
            # Slice fill part
            fill_lower_bound = MIN_TIME + i * (MAX_TIME - MIN_TIME) / YEAR_SLICE_COUNT
            fill_upper_bound = MIN_TIME + (i + 1) * (MAX_TIME - MIN_TIME) / YEAR_SLICE_COUNT

            fill_values = nan_values.loc[fill_lower_bound <= nan_values["timestamp"]]
            fill_values = fill_values.loc[fill_values["timestamp"] <= fill_upper_bound]

            # Slice training values
            training_lower_bound = MIN_TIME + i * (MAX_TIME - MIN_TIME) / YEAR_SLICE_COUNT - WINDOW_OVERLAP * 30 * 24 * 60 * 60
            training_upper_bound = MIN_TIME + (i + 1) * (MAX_TIME - MIN_TIME) / YEAR_SLICE_COUNT + WINDOW_OVERLAP * 30 * 24 * 60 * 60

            training_values = not_nan_values.loc[training_lower_bound <= not_nan_values["timestamp"]]
            training_values = training_values.loc[training_values["timestamp"] <= training_upper_bound]

            # Fit model
            params, _ = curve_fit(model_station, training_values["timestamp"], training_values["Valeur"], maxfev=100_000)

            # Save data
            timestamps = fill_values["timestamp"]
            values = model_station(timestamps, *params)
            data[data["idPolair"] == station_id].iloc[nan_indices]["Valeur"] = values 
        
        fig = plt.figure()
        fig.scatter(data[data["idPolair"] == station_id]["timestamp"], data[data["idPolair"] == station_id]["Valeur"], label=f"{station_id}")
        fig.scatter(nan_values["timestamp"], nan_values["Valeur"], label=f"{station_id} pred", color="red")
        fig.plot()

        break

    return data
