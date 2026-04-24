import requests
from datetime import datetime
from gaussianPuff.config import WindType, PasquillGiffordStability
import numpy as np

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_meteo(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "windspeed_10m",
            "winddirection_10m",
            "relativehumidity_2m",
            "is_day",
            "cloud_cover"
        ],
        "timezone": "UTC"
    }

    response = requests.get(OPEN_METEO_URL, params=params)
    response.raise_for_status()

    data = response.json()
    hourly = data["hourly"]

    # ora corrente in UTC, arrotondata all’ora
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

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
        "wind_speed": hourly["windspeed_10m"][time_index],
        "wind_dir": hourly["winddirection_10m"][time_index],
        "RH": hourly["relativehumidity_2m"][time_index],
        "is_day": hourly["is_day"][time_index],
        "cloud_cover": hourly["cloud_cover"][time_index],
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


def infer_dry_size_from_openmeteo(wind_speed: float) -> float:
    """
    Stima della dimensione secca delle particelle (µm)
    basata solo su condizioni atmosferiche.
    """

    # Valore tipico aerosol urbano di fondo
    dry_size = 0.6  # µm

    # Più vento → particelle dominanti più piccole
    if wind_speed >= 8.0:
        dry_size = 0.3
    elif wind_speed >= 5.0:
        dry_size = 0.4
    elif wind_speed >= 3.0:
        dry_size = 0.5

    return dry_size

def infer_stability_from_openmeteo(
    wind_speed: float,
    is_day: bool,
    cloud_cover: float
) -> PasquillGiffordStability:

    # Giorno
    if is_day:
        if wind_speed < 2.0:
            return PasquillGiffordStability.VERY_UNSTABLE      # A
        elif wind_speed < 3.5:
            return PasquillGiffordStability.MODERATELY_UNSTABLE  # B
        elif wind_speed < 5.0:
            return PasquillGiffordStability.SLIGHTLY_UNSTABLE  # C
        else:
            return PasquillGiffordStability.NEUTRAL            # D

    # Notte
    else:
        if wind_speed < 2.0 and cloud_cover < 30:
            return PasquillGiffordStability.VERY_STABLE        # F
        elif wind_speed < 3.5:
            return PasquillGiffordStability.MODERATELY_STABLE  # E
        else:
            return PasquillGiffordStability.NEUTRAL            # D
