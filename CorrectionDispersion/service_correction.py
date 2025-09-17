import torch
import numpy as np
from CorrectionDispersion.MCxM import MCxM_CNN
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(binary_map, m=500, device=None, pretrained_path=None):
    model_path = os.path.join(SCRIPT_DIR, "models", "mcxm_cnn_model.pth")

    loaded_model = MCxM_CNN(binary_map, m=m, n_channel=1, n_mask_correction=3, wind_dim=2, n_global_features=0).to(device)

    # Carica i pesi
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()
    return loaded_model


def correct_dispersion(wind_dir, wind_speed, concentration_map, building_map, global_feature=None, device=None, m=500, pretrained_path=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(building_map, m=m, device=device, pretrained_path=pretrained_path)

    mc = torch.tensor(concentration_map, dtype=torch.float32, device=device).unsqueeze(0)  # (1, m, m)

    # Converti wind_dir in caratteristiche di vento (coseno e seno)
    wind_dir_cos=np.cos(np.radians(wind_dir))
    wind_dir_sin=np.sin(np.radians(wind_dir))
    wind_dir_angle = np.atan2(wind_dir_cos, wind_dir_sin)

    wind_features = torch.tensor([[wind_dir_angle, wind_speed]], dtype=torch.float32, device=device)

    if global_feature is not None:
        global_features = torch.tensor(global_feature, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        global_features = None
        
    # --- Inference ---
    with torch.no_grad():
        output = model(mc, wind_features, global_features)
        output = output.detach().cpu().numpy()[0]

    return output