from pandas import read_csv
from numpy import unique

train_data = read_csv("data/computed/train_data.csv")
test_data = read_csv("data/computed/test_data_1.csv")

ID_TO_KEEP_COUNT = 3
station_ids = unique(train_data["idPolair"])
station_ids_to_keep = station_ids[:ID_TO_KEEP_COUNT]

new_train_data = train_data[train_data["idPolair"].isin(station_ids_to_keep)]
new_train_data.to_csv("data/computed/reduced_train_data.csv")

new_test_data = test_data[test_data["idPolair"].isin(station_ids_to_keep)]
new_test_data.to_csv("data/computed/reduced_test_data.csv")
