from __future__ import annotations

from functools import cache

from pandas import DataFrame
from numpy import unique, min, max, any
from datetime import datetime, timedelta


def has_alert_been_raised_next_day(data: DataFrame, today_timestamp: float, day_forecast: int = 1) -> bool:
    today = datetime.fromtimestamp(today_timestamp)
    tomorrow = today + timedelta(days=1)
    tomorrow_start = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0)
    tomorrow_end = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 23, 59, 0)

    data = data[tomorrow_start.timestamp() <= data["timestamp"]]
    data = data[data["timestamp"] <= tomorrow_end.timestamp()]
    return any(data["alerte"])

@cache
def format_dataset(data: DataFrame, hours: list[int], day_forecast: int = 1) -> DataFrame:
    """Pack data by batch with values of previous 'hours' and alerte 'day_forecast' later.

    For instance, if hours is [0, -1, -2], and day_forecast is 1
    the returned DataFrame will have columns 'Valeur_h0', 'timestamp_h0', 'Valeur_h-1', 'timestamp_h-1', 'Valeur_h-2', 'timestamp_h-2', 'alerte_d+1', 'Valeur_d+1'

    h0 is set to midday and alerte_d+day_forecast is True if an alert occurs during the day d+day_forecast
    """
    print(f"Formatting dataset : hours={hours}, forecast={day_forecast} day{'s' if abs(day_forecast) > 1 else ''}")

    hour_offset = min(hours)
    min_time = min(data["timestamp"])
    max_time = max(data["timestamp"])
    min_datetime = datetime.fromtimestamp(min_time)
    start_time = datetime(min_datetime.year, min_datetime.month, min_datetime.day, 12, 0, 0)

    while (start_time - timedelta(hours=int(hour_offset))).timestamp() < min_time:
        start_time += timedelta(days=1)

    columns = list(data.columns)
    columns.remove("timestamp")
    columns.remove("Valeur")
    columns.remove("alerte")
    for hour in hours:
        columns.append(f"timestamp_h{hour}")
        columns.append(f"Valeur_h{hour}")
    columns.append(f"alerte_d+{day_forecast}")
    columns.append(f"Valeur_d+{day_forecast}")

    new_data = DataFrame(columns=columns)

    for station_id in unique(data["idPolair"]):
        current_time = start_time
        station_values = data[data["idPolair"] == station_id]

        while current_time.timestamp() <= max_time:

            row_h0 = station_values[station_values["timestamp"] == current_time.timestamp()]
            new_row = [
                row_h0["Organisme"],
                row_h0["Station"],
                row_h0["idPolair"],
            ]

            for hour in hours:
                row_h_plus_hour = station_values[station_values["timestamp"] == (current_time + timedelta(hours=hour)).timestamp()]

                new_row += [
                    row_h_plus_hour["timestamp"],
                    row_h_plus_hour["Valeur"],
                ]

            row_d_plus_day_forecast = station_values[station_values["timestamp"] == (current_time + timedelta(days=day_forecast)).timestamp()]
            new_row += [
                has_alert_been_raised_next_day(station_values, current_time.timestamp()),
                row_d_plus_day_forecast["Valeur"]
            ]

            new_data.loc[len(new_data)] = [len(new_data)] + new_row

            current_time += timedelta(days=1)

    print("Dataset formatted")
    return new_data
