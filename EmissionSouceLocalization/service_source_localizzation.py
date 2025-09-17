# service_source_localizzation.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import joblib
from typing import List
import os

class SensorTimeSeries:
    def __init__(self, time, conc, simulation_id, wind_dir_x, wind_dir_y, wind_speed, wind_type):
        self.time = time
        self.conc = conc
        self.simulation_id = simulation_id
        self.wind_dir_x = wind_dir_x
        self.wind_dir_y = wind_dir_y
        self.wind_speed = wind_speed
        self.wind_type = wind_type

    def dict(self):
        return {
            "time": self.time,
            "conc": self.conc,
            "simulation_id": self.simulation_id,
            "wind_dir_x": self.wind_dir_x,
            "wind_dir_y": self.wind_dir_y,
            "wind_speed": self.wind_speed,
            "wind_type": self.wind_type
        }

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "emission_source_model.pkl")

model = joblib.load(MODEL_PATH)

def estrai_feature(time, conc, window=None, spike_height=None):
   
    time = np.array(time)
    conc = np.array(conc)

    # Concentrazione massima e tempo del massimo
    C_max = np.max(conc)
    idx_max = np.argmax(conc)
    t_peak = time[idx_max]

    # Primo picco
    if spike_height is None:
        spike_height = 0.1 * C_max
    peaks, _ = find_peaks(conc, height=spike_height)
    t_first_peak = time[peaks[0]] if len(peaks) > 0 else np.nan

    # Rise / fall rate
    rise_rate = (C_max - conc[0]) / (t_peak - time[0] + 1e-6)
    fall_rate = (C_max - conc[-1]) / (time[-1] - t_peak + 1e-6)

    # Media e deviazione standard
    if window is not None:
        mask = (time >= window[0]) & (time <= window[1])
        conc_window = conc[mask]
    else:
        conc_window = conc
    mean_val = np.mean(conc_window)
    std_val = np.std(conc_window)

    # Area sotto la curva
    auc = np.trapezoid(conc, time)

    # Durata plume
    above = np.where(conc > spike_height)[0]
    if len(above) > 0:
        t_plume_start = time[above[0]]
        t_plume_end = time[above[-1]]
        plume_duration = t_plume_end - t_plume_start
    else:
        plume_duration = 0.0

    # Spike count / frequency
    spike_count = len(peaks)
    total_duration = time[-1] - time[0]
    spike_freq = spike_count / total_duration if total_duration > 0 else 0.0

    return {
        "C_max": C_max,
        "t_peak": t_peak,
        "t_first_peak": t_first_peak,
        "mean": mean_val,
        "std": std_val,
        "AUC": auc,
        "rise_rate": rise_rate,
        "fall_rate": fall_rate,
        "plume_duration": plume_duration,
        "spike_count": spike_count,
        "spike_frequency": spike_freq
    }

def predict_source(sensors):
    
    df = pd.DataFrame([s.dict() for s in sensors])
    agg_features_list = []

    for sim_id, group in df.groupby("simulation_id"):
        valid_sensors = group[group["conc"].apply(lambda x: len(x) > 0)]
        sensori_features = []

        for _, row in valid_sensors.iterrows():
            feat = estrai_feature(row["time"], row["conc"])
            sensori_features.append(feat)

        if sensori_features:
            df_features = pd.DataFrame(sensori_features)
            agg_features = df_features.mean().add_suffix("_mean").to_dict()
            agg_features.update(df_features.std().add_suffix("_std").to_dict())
        else:
            # Nessun sensore valido → NaN
            agg_features = {col: np.nan for col in [
                "C_max_mean","t_peak_mean","t_first_peak_mean","mean_mean","std_mean",
                "AUC_mean","rise_rate_mean","fall_rate_mean","plume_duration_mean",
                "spike_count_mean","spike_frequency_mean"]}
        
        first_row = group.iloc[0]
        agg_features.update({
            "wind_dir_x": first_row["wind_dir_x"],
            "wind_dir_y": first_row["wind_dir_y"],
            "wind_speed": first_row["wind_speed"],
            "wind_type": first_row["wind_type"]
        })
        agg_features_list.append(agg_features)

    X_input = pd.DataFrame(agg_features_list)

    y_pred = model.predict(X_input)
    result = [{"source_x": float(x), "source_y": float(y)} for x, y in y_pred]

    return {"predicted_location": result}