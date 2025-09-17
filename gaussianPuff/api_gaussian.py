from fastapi import FastAPI
import os, sys
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import logging
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from gaussianPuff.gaussianModel import run_dispersion_model
from gaussianPuff.config import ModelConfig
import uvicorn

# Configurazione logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class ModelConfigRequest(BaseModel):
    days: int
    RH: float
    aerosol_type: str
    humidify: bool
    stability_profile: str
    stability_value: str
    wind_type: str
    wind_speed: float
    output: str
    stacks: List[Tuple[float, float, float, float]]
    dry_size: Optional[float] = 60e-9
    x_slice: Optional[int] = 26
    y_slice: Optional[int] = 1
    grid_size: Optional[int] = 500
    dispersion_model: str
    config_puff: Optional[dict] = None

class Payload(BaseModel):
    model_config: ModelConfigRequest
    bounds: List[float] = Field(..., min_items=4, max_items=4)

app = FastAPI()

@app.post("/start_simulation")
def start_simulation(payload: dict):
    logger.info("Ricevuta richiesta /start_simulation")
    try:
        config = ModelConfig(**payload.get("model_config", {}))
        bounds = payload.get("bounds", None)
        logger.info(f"Configurazione modello creata: {config}")
        logger.info(f"Bounds ricevuti: {bounds}")

        result = run_dispersion_model(config, bounds)
        logger.info("Simulazione completata")

        C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = result

        response = {
            "status": "success",
            "concentration": C1.tolist() if isinstance(C1, np.ndarray) else C1,
            "x": x.tolist() if isinstance(x, np.ndarray) else x,
            "y": y.tolist() if isinstance(y, np.ndarray) else y,
            "z": z.tolist() if isinstance(z, np.ndarray) else z,
            "times": times.tolist() if isinstance(times, np.ndarray) else times,
            "stability": stability.tolist() if isinstance(stability, np.ndarray) else stability,
            "wind_dir": wind_dir,
            "stab_label": stab_label,
            "wind_label": wind_label,
            "puff": puff if puff is not None else None
        }

        return response

    except Exception as error:
        logger.exception("Errore durante la simulazione")
        raise error

"""if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)"""
