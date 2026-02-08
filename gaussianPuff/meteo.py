import requests
from datetime import datetime
from gaussianPuff.config import WindType
import numpy as np

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_meteo(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "winddirection_10m",
        ],
        "timezone": "UTC"
    }

    response = requests.get(OPEN_METEO_URL, params=params)
    response.raise_for_status()

    data = response.json()
    hourly = data["hourly"]

    # ora corrente in UTC, arrotondata all’ora
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    # parse robusto degli orari API
    times = [
        datetime.fromisoformat(t.replace("Z", ""))
        for t in hourly["time"]
    ]

    # trova l’ora più vicina (niente ValueError)
    time_index = min(
        range(len(times)),
        key=lambda i: abs(times[i] - now)
    )

    return {
        "wind_dir": hourly["winddirection_10m"][time_index],
    }


def infer_wind_type_from_openmeteo(wind_dir: np.ndarray) -> WindType:
    """
    Deduce WindType from a time series of wind directions (degrees).
    """
    wind_dir = np.asarray(wind_dir)

    # Convert degrees → radians
    theta = np.radians(wind_dir)

    mean_cos = np.mean(np.cos(theta))
    mean_sin = np.mean(np.sin(theta))

    R = np.sqrt(mean_cos**2 + mean_sin**2)

    if R > 0.9:
        return WindType.CONSTANT
    elif R > 0.6:
        return WindType.PREVAILING
    else:
        return WindType.FLUCTUATING