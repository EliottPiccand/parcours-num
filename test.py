from pandas import read_csv
from cleaning_methods.guesstimate_sin import tmp

data = read_csv("data/computed/guesstimate_sin.csv")
data = tmp(data)
data.to_csv("data/computed/guesstimate_sin_2.csv")