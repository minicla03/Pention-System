from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import sys
import logging
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from EmissionSouceLocalization import service_source_localizzation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

"""class SensorTimeSeries(BaseModel):
    simulation_id: str
    sensor_id: str
    time: List[float]
    conc: List[float]
    wind_dir_x: float
    wind_dir_y: float
    wind_speed: float
    wind_type: str"""

app = FastAPI()

@app.post("/predict_source_raw")
def predict_source_raw(sensors):

    logger.info(f"Ricevuta richiesta /predict_source_raw con {len(sensors)} sensori")

    try:
        result = service_source_localizzation.predict_source(sensors)
        logger.info("Predizione sorgente completata")

        return result.get("predicted_location", (None, None))

    except Exception as e:
        logger.exception("Errore durante la predizione della sorgente")
        raise e

"""if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)"""
