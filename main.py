from __future__ import annotations

from datetime import timedelta
from itertools import product
from pathlib import Path
from time import perf_counter_ns as get_time_ns
from typing import TYPE_CHECKING

from pandas import DatetimeIndex, read_csv
from pyreadr import read_r
from numpy import int64
import matplotlib.pyplot as plt

from cleaning_methods import CLEANING_METHODS
from prediction_methods import PREDICTION_METHODS

if TYPE_CHECKING:
    from pandas import DataFrame


DATA_DIR = Path("data")
def load_rds(filename: str) -> DataFrame:
    print(f"- Loading data from file '{DATA_DIR / filename}'")
    return read_r(DATA_DIR / filename)[None]

def default_data_filter(data: DataFrame) -> DataFrame:
    """Remove column 'Mesure' and replace 'date' by 'timestamp'"""

    data.index = DatetimeIndex(data["date"]).tz_localize("UTC")
    data["timestamp"] = data.index.values.astype(int64) // 1e9
    data = data.drop(columns=["date"])
    data = data.drop(columns=["Mesure"])
    return data

TRAIN_DATA = load_rds("train_data.rds") # ['Organisme', 'Station', 'Mesure', 'Valeur', 'idPolair', 'date']
TEST_DATA = load_rds("test_data_1.rds")

TRAIN_DATA = default_data_filter(TRAIN_DATA) # ['Organisme', 'Station', 'Valeur', 'idPolair', 'timestamp']

def display_time_diff_to_now(start: int) -> str:
    return str(timedelta(seconds=(get_time_ns() - start) / 1e9))

min_error = float("inf")
min_error_arg = None
start_time = get_time_ns()
for (cleaning_method_name, cleaning_method), (prediction_method_name, train_model, get_model_error) in product(CLEANING_METHODS, PREDICTION_METHODS):
    print()
    print("Cleaning method   :", cleaning_method_name)
    print("Prediction method :", prediction_method_name)

    method_start_time = get_time_ns()

    cleaned_data_path = Path(f"data/computed/{cleaning_method_name.replace(' ', '_')}.csv")
    cleaned_data_path.parent.mkdir(exist_ok=True)
    if cleaned_data_path.exists():
        print("- Loading cleaned train data")
        cleaned_train_data = read_csv(cleaned_data_path)
    else:
        print("- Computing cleaned train data")
        cleaned_train_data = cleaning_method(TRAIN_DATA)
        print("- Saving cleaned train data")
        cleaned_train_data.to_csv(cleaned_data_path)
    import numpy as np
    print(np.any(TRAIN_DATA['Valeur'].isna()))

    print(len(TRAIN_DATA), len(cleaned_train_data))

    print(np.any(cleaned_train_data['Valeur'].isna()))

    plt.show()

    break

    # model = train_model(cleaned_train_data)

    # error = get_model_error(model, TEST_DATA)
    # if error < min_error:
    #     min_error = error
    #     min_error_arg = (cleaning_method_name, prediction_method_name)

    # print(f"Compute time      : {display_time_diff_to_now(method_start_time)}")
    # print(f"> error : {error:.3f}%")

# print()
# print()
# print()
# print(f"Total Compute Time : {display_time_diff_to_now(start_time)}")
# print("Best methods : ")
# print("- Cleaning method   :", min_error_arg[0])
# print("- Prediction method :", min_error_arg[1])
# print("- Error             :", min_error)
