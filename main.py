from __future__ import annotations

from datetime import timedelta
from itertools import product
from pathlib import Path
from time import perf_counter_ns as get_time_ns
from typing import TYPE_CHECKING

from pandas import DatetimeIndex, read_csv
from pyreadr import read_r
from numpy import int64

from cleaning_methods import CLEANING_METHODS
from prediction_methods import PREDICTION_METHODS

if TYPE_CHECKING:
    from pandas import DataFrame


DATA_DIR = Path("data")
COMPUTED_DATA_DIR = DATA_DIR / "computed"

# Training & Testing data
def load_rds(filename: Path) -> DataFrame:
    return read_r(filename)[None]

def default_data_filter(data: DataFrame) -> DataFrame:
    """Remove column 'Mesure' and replace 'date' by 'timestamp'"""

    # data.index = DatetimeIndex(data["date"]).tz_localize("UTC")
    data["timestamp"] = DatetimeIndex(data["date"]).tz_localize("UTC").values.astype(int64) // 1e9
    data = data.drop(columns=["date", "Mesure"])
    data["alerte"] = data["Valeur"] > 100
    return data

TRAIN_DATA_PATH = DATA_DIR / "train_data.rds"
SAVED_TRAIN_DATA_PATH = COMPUTED_DATA_DIR / "train_data.csv"
if SAVED_TRAIN_DATA_PATH.exists():
    print("Loading training data")
    TRAIN_DATA = read_csv(SAVED_TRAIN_DATA_PATH)
else:
    print("Filtering training data")
    TRAIN_DATA = load_rds(TRAIN_DATA_PATH) # ['Organisme', 'Station', 'Mesure', 'Valeur', 'idPolair', 'date']
    TRAIN_DATA = default_data_filter(TRAIN_DATA) # ['Organisme', 'Station', 'Valeur', 'idPolair', 'timestamp']
    print("Saving filtered training data")
    SAVED_TRAIN_DATA_PATH.parent.mkdir(exist_ok=True)
    TRAIN_DATA.to_csv(SAVED_TRAIN_DATA_PATH)

TEST_DATA = load_rds(DATA_DIR / "test_data_1.rds")

def display_time_diff_to_now(start: int) -> str:
    return str(timedelta(seconds=(get_time_ns() - start) / 1e9))

# Compare cleaning and prediction methods
min_error = float("inf")
min_error_arg = None
start_time = get_time_ns()
for (cleaning_method_name, cleaning_method), (prediction_method_name, train_model, get_model_error) in product(CLEANING_METHODS, PREDICTION_METHODS):
    print()
    print("Cleaning method   :", cleaning_method_name)
    print("Prediction method :", prediction_method_name)

    method_start_time = get_time_ns()

    cleaned_data_path =  COMPUTED_DATA_DIR / f"{cleaning_method_name.replace(' ', '_')}.csv"
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

    # error = get_model_error(model, TEST_DATA)
    # if error < min_error:
    #     min_error = error
    #     min_error_arg = (cleaning_method_name, prediction_method_name)

    print(f"Compute time      : {display_time_diff_to_now(method_start_time)}")
    # print(f"> error : {error:.3f}%")

print()
print()
print()
print(f"Total Compute Time : {display_time_diff_to_now(start_time)}")
# print("Best methods : ")
# print("- Cleaning method   :", min_error_arg[0])
# print("- Prediction method :", min_error_arg[1])
# print("- Error             :", min_error)

while True:
    pass
