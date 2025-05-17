from __future__ import annotations

from os import getenv
from typing import TYPE_CHECKING

from numpy import array
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from .format_dataset import format_dataset, has_alert_been_raised_next_day

if TYPE_CHECKING:
    from numpy import ndarray


class Predictor[T]:

    def __init__(self, predictor_class: T, is_regressor: bool, hours: list[int], cross_validation_params: dict, reduced_cross_validation_params: dict) -> None:
        self.predictor_class = predictor_class
        self.input_features = [
            f"timestamp_h{hour}"
            for hour in hours
        ] + [
            f"Valeur_h{hour}"
            for hour in hours
        ]
        self.hours = hours
        self.cross_validation_params = cross_validation_params
        self.reduced_cross_validation_params = reduced_cross_validation_params
        self.is_regressor = is_regressor
    
    def tune_hyperparameters(self, x: ndarray, y: ndarray) -> dict:
        if getenv("USE_REDUCED_DATA", "False") == "True":
            param_grid = self.reduced_cross_validation_params
        else:
            param_grid = self.cross_validation_params

        grid_search = GridSearchCV(self.predictor_class(), param_grid, cv=5, n_jobs=-1)
        grid_search.fit(x, y)

        return grid_search.best_params_

    def train(self, data: DataFrame) -> T:
        data = format_dataset(data, self.hours)

        x = data[self.input_features].to_numpy()
        if self.is_regressor:
            y = data["Valeur_d+1"].to_numpy()
        else:
            y = data["alerte_d+1"].to_numpy()

        print("   - Performing cross validation")
        best_params = self.tune_hyperparameters(x, y)
        print("     > Best params :", best_params)

        print("   - Fitting best model found")
        model = self.predictor_class(**best_params)
        model.fit(x, y)

        return model
    
    def get_error(self, model: T, data: DataFrame) -> float:
        data = format_dataset(data, self.hours)
    
        x = data[self.input_features]
        
        print("   - Computing real values")
        y = array([
            has_alert_been_raised_next_day(data, ts)
            for ts in x["timestamp_h0"]
        ])
        x = x.to_numpy()

        print("   - Predicting values")
        y_pred = model.predict(x)

        if self.is_regressor:
            y_pred = y_pred > 100

        print("   - Computing accuracy")
        return 1 - accuracy_score(y, y_pred)
