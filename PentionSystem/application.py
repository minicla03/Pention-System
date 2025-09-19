import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
import requests
import matplotlib.pyplot as plt
import sys
import os
import folium
from windrose import WindroseAxes
from folium.plugins import HeatMap
from collections import Counter

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from gaussianPuff.Sensor import SensorSubstance, SensorAir
from gaussianPuff.config import NPS, OutputType, DispersionModelType, ModelConfig

API_URL = "http://127.0.0.1:"

nps_classes = [
    'Cathinone analogues',
    'Cannabinoid analogues',
    'Phenethylamine analogues',
    'Piperazine analogues',
    'Tryptamine analogues',
    'Fentanyl analogues'
]

def random_position(free_cells):
    idx = np.random.choice(len(free_cells))
    y, x = free_cells[idx]
    return float(y), float(x)

def plot_dispersion_on_map(min_lat, min_lon, max_lat, max_lon, sensors, dispersion_map, source_lat=None, source_lon=None, 
                           title="Mappa Dispersione", wind_dir=None, wind_speed=None, puff_list=None, stability_class=1, n_show=10):
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="cartodbpositron")

    # Sensori
    for s in sensors:
        folium.Marker(
            [s.y, s.x],
            popup=f"Sensor {s.id}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    # Sorgente stimata
    if source_lat is not None and source_lon is not None:
        folium.Marker(
            [source_lat, source_lon],
            popup="Sorgente Stimata",
            icon=folium.Icon(color="red", icon="fire")
        ).add_to(m)

    # Heatmap della dispersione
    heat_data = []
    rows = len(dispersion_map)
    cols = len(dispersion_map[0]) if rows > 0 else 0

    for i in range(rows):
        for j in range(cols):
            lat = min_lat + (max_lat - min_lat) * (i / max(rows-1, 1))
            lon = min_lon + (max_lon - min_lon) * (j / max(cols-1, 1))
            conc = dispersion_map[i][j]
            if conc > 0:  # evita punti nulli
                heat_data.append([lat, lon, conc])

    if heat_data:
        HeatMap(
            heat_data,
            radius=25,
            blur=15,
            max_val=max([c[2] for c in heat_data]),
            min_opacity=0.2
        ).add_to(m)

    title_html = f'''
         <h3 align="center" style="font-size:18px"><b>{title}</b></h3>
         '''
    m.get_root().html.add_child(folium.Element(title_html))

    return m

def plot_plan_view(C1, x, y, title="Plan View", puff_list=None, stability_class=1):
    with dispersion_placeholder:

        fig, ax_main = plt.subplots(figsize=(8, 6))

        # Integra la concentrazione nel tempo lungo l'asse 2 (T)
        data = np.trapezoid(C1, axis=2) * 1e6  # µg/m³ #type:ignore
        vmin = np.percentile(data, 5)
        vmax = np.percentile(data, 95)

        # Plot della concentrazione integrata
        pcm = ax_main.pcolor(x, y, data, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, ax=ax_main, label=r'$\mu g \cdot m^{-3}$')
        ax_main.set_xlabel('x (m)')
        ax_main.set_ylabel('y (m)')
        ax_main.set_title(title)
        ax_main.axis('equal')

        st.pyplot(fig, clear_figure=True)

def plot_wind_rose(wind_dir, wind_speed):

    with wind_rose_placeholder:
        if wind_dir is not None and wind_speed is not None:

            fig = plt.figure(figsize=(6, 6))

            #inset_pos = [0.65, 0.65, 0.3, 0.3]  # left, bottom, width, height in figure coords
            ax_inset = WindroseAxes(fig)
            fig.add_axes(ax_inset)

            # Plot rosa dei venti con direzioni e velocità
            wind_dir = np.array(wind_dir) % 360
            wind_speed = np.full_like(wind_dir, fill_value=wind_speed, dtype=float)
            ax_inset.bar(wind_dir, wind_speed, normed=True, opening=0.8, edgecolor='white')
            ax_inset.set_legend(loc='lower right', title='Wind speed (m/s)')
            ax_inset.set_title("Rosa dei venti")

            st.pyplot(fig, clear_figure=True)

def plot_binary_map(binary_map, metadata, sensors=None):
    with map_section:

        fig, ax = plt.subplots(figsize=(8, 8))
    
        x_min, y_min, x_max, y_max = metadata['bounds']
        im = ax.imshow(binary_map, cmap='gray', extent=(x_min, x_max, y_min, y_max), origin='lower')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        building_cells = np.sum(binary_map == 0)
        free_cells = np.sum(binary_map == 1)
        
        info_text = f"""
        Griglia: {metadata.get('grid_size', 'N/A')}×{metadata.get('grid_size', 'N/A')}
        Edifici totali: {metadata.get('total_buildings', 'N/A')}
        Celle edifici: {building_cells:,}
        Celle libere: {free_cells:,}
        CRS: {metadata.get('crs', 'N/A')}
        Risoluzione: {metadata.get('resolution (m)', 'N/A')} m
        Densità edifici: {metadata.get('building_density', 'N/A'):.1f}%
        Altezza media edifici: {metadata.get('mean_height', 'N/A')} m
        Coordinate origine: {metadata.get('origin', 'N/A')}
        Città: {metadata.get('city', 'N/A')}
        """
        
        ax.text(0.02, -0.08, info_text, fontsize=10, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        ax.set_xlabel("Coordinate X (grid)")
        ax.set_ylabel("Coordinate Y (grid)")
        ax.grid(False)

        if sensors is not None:
            for s in sensors:
                x = s.x
                y = s.y
                faulty = s.is_fault

                if not faulty:
                    ax.scatter(x, y, c="green", marker="o", edgecolors="black", s=80, label="Operating")
                else:
                    ax.scatter(x, y, c="red", marker="X", edgecolors="black", s=100, label="Faulty")

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Occupazione")
        
        st.pyplot(fig, clear_figure=True)

def run_application(payload):
    
    n_sensors = payload.get("Number of sensors", 10)
    payload.pop("Number of sensors", None)

    progress = 0
    progress_bar.progress(progress)

    # --- Binary map generation
    status_text.text("Binary map generation...")

    response = requests.post(f"{API_URL}8001/generate_binary_map", json=payload)
    if response.status_code != 200:
        st.error("Error in binary map generation.")
        return None
    
    data = response.json()
    if data.get("status_code") != "success":
        st.error("Error in binary map generation.")
        return None
    
    binary_map = np.array(data.get("map"), dtype=np.float32)
    metadata = data.get("metadata", {})
    free_cells = np.argwhere(binary_map == 1)

    progress += 20
    progress_bar.progress(progress)

    # --- Meteo condition
    status_text.text("Sample meteo condition...")
    sensor_air = SensorAir(sensor_id=00, x=0.0, y=0.0, z=2.0)
    wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH = sensor_air.sample_meteorology()

    if weather_section is not None:
        weather_placeholder.markdown(
            f"💨 **Wind speed (m/s):** {wind_speed}  \n"
            f"💨 **Wind type:** {wind_type.value}  \n"
            f"📈 **Stability:** {stability_type.value}  \n"
            f"♒︎ **Relative Humidity (%):** {RH}"
        ) 

    # --- Sensor substance
    status_text.text("Air sampling...")
    sensors_substance = []
  
    for i in range(n_sensors):
        x, y = random_position(free_cells)
        sensor_substance = SensorSubstance(i, x=x, y=y, z=2.0,
                                           noise_level=round(np.random.uniform(0.0, 0.0005), 4))
        sensors_substance.append(sensor_substance)

    plot_binary_map(binary_map, metadata, sensors_substance)

    mass_spectrum = []
    for sensor in sensors_substance:
        recording = sensor.run_sensor(wind_speed, stability_type, RH, wind_type)
        recording = [rec for rec in recording if not np.isnan(rec).any()]
        mass_spectrum.extend(recording)

    print(f"1->{type(mass_spectrum)}") # list
    print(f"2->{type(mass_spectrum[0])}") # numpy.ndarray
       
    if sensors_section is not None:
        sensor_info = [{"ID": s.id, "x": s.x, "y": s.y, "Status": "Operating" if not s.is_fault else "Faulty",} 
                       for s in sensors_substance]
        sensors_placeholder.table(sensor_info)

    progress += 20
    progress_bar.progress(progress) 

    # --- NPS classification
    status_text.text("NPS classification...")
    substance_nps = []

    if mass_spectrum:
        spectra_json = [m.tolist() for m in mass_spectrum]
        print(f"spectra_json: {type(spectra_json)}")
        response_dnn = requests.post(f"{API_URL}8000/predict_dnn", json={"spectra": spectra_json})

        if response_dnn.status_code == 200:
            predictions = response_dnn.json().get("predictions", [])
            print(len(predictions))
            substance_nps = [pred for pred in predictions if pred in nps_classes]
        else:
            st.error(f"Errore API {response_dnn.status_code}")

    print(type(substance_nps))
    print(len(substance_nps))

    if substance_nps:
        most_common_substance = Counter(substance_nps).most_common(1)[0][0]
        print(most_common_substance)
        nps = NPS.from_string(most_common_substance)
    else:
        most_common_substance = None
        print("Nessuna sostanza presente")

    if nps_section is not None:
        if substance_nps:
            nps_placeholder.markdown(most_common_substance)
        else:
            nps_placeholder.warning("No NPS identified.")

    progress += 20
    progress_bar.progress(progress)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    x_src, y_src = random_position(free_cells)
    h_src = round(np.random.uniform(1, 10), 2)  # altezza del pennacchio
    Q = round(np.random.uniform(0.0001, 0.01), 4)  # tasso di emissione
    stacks = [(x_src, y_src, Q, h_src)]

    print(stability_value)
    print(wind_speed)
    print(wind_type)

    param_gaussian_model = ModelConfig(
        days=10,
        RH=RH,
        aerosol_type=NPS(nps),
        humidify=humidify,
        stability_profile=stability_type,
        stability_value=stability_value,
        wind_type=wind_type,
        wind_speed=wind_speed,
        output=OutputType.PLAN_VIEW,
        stacks=stacks,
        dry_size=dry_size, x_slice=26, y_slice=1,
        dispersion_model=DispersionModelType.PLUME)

    bounds = (payload["min_lon"], payload["min_lat"], payload["max_lon"], payload["max_lat"])

    response_gauss = requests.post(f"{API_URL}8002/start_simulation",
                                   json={"config": param_gaussian_model.to_dict(),
                                         "bounds": bounds})

    print("risposta ottenuta")
    print(f"code: {response_gauss.status_code}")
    print(response_gauss)

    if response_gauss.status_code != 200:
        st.error("Error in Gaussian puff simulation 01.")
        return sensors_substance, substance_nps, None, None, None, metadata

    gauss_data = response_gauss.json()
    x_raw = gauss_data.get("x", [])
    y_raw = gauss_data.get("y", [])
    times_raw = gauss_data.get("times", [])
    wind_dir_raw = gauss_data.get("wind_dir")
    C1_raw = gauss_data.get("concentration", [])

    x=np.array(x_raw)
    y=np.array(y_raw)
    times=np.array(times_raw)
    wind_dir=np.array(wind_dir_raw)
    C1=np.array(C1_raw)

    print(type(C1))
    print(C1.shape)
    print(type(wind_dir))
    print(wind_dir.shape)
    print(type(wind_speed))
    print(type(x))
    print(x.shape)
    print(type(y))
    print(y.shape)

    status_text.text("Dispersion map generation...")
    plot_plan_view(C1, x, y)
    status_text.text("Wind rose graph generation...")
    plot_wind_rose(wind_dir, wind_speed) #okk

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # --- Localizzazione sorgente
    status_text.text("Source estimation...")

    payload_sensors = []
    for s in sensors_substance:
        if not s.is_fault:
            s.sample_substance(C1, x, y, times)
            # s.sample_substance_synthetic()  # se serve

            sensor_data = {
                "sensor_id": s.id,
                "time": [],
                "conc": [],
                "wind_dir_x": [],
                "wind_dir_y": [],
                "wind_speed": [],
                "wind_type": wind_type.value
            }

            for idx, (t_idx, conc) in enumerate(zip(s.times, s.noisy_concentrations)):
                if idx >= len(wind_dir):
                    break
                wd = wind_dir[idx]

                sensor_data["time"].append(t_idx)
                sensor_data["conc"].append(float(conc) if not s.is_fault else float("nan"))
                sensor_data["wind_dir_x"].append(float(np.cos(np.deg2rad(wd))) if not s.is_fault else float("nan"))
                sensor_data["wind_dir_y"].append(float(np.sin(np.deg2rad(wd))) if not s.is_fault else float("nan"))
                sensor_data["wind_speed"].append(float(wind_speed) if not s.is_fault else float("nan"))

            payload_sensors.append(sensor_data)

    status_text.text("Start the prediction of the source...")
    response = requests.post(f"{API_URL}8003/predict_source_raw", json=payload_sensors)
    origin_lat = response.json().get("source_x", None)
    origin_lon = response.json().get("source_y", None)

    if source_section is not None:
        if origin_lat is not None and origin_lon is not None:
            source_placeholder.markdown(f"Lat: {origin_lat}, Long: {origin_lon}")
        else:
            source_placeholder.warning("Source not estimated.")

    progress += 20
    progress_bar.progress(progress)

    # --- gaussian plume dispersion (raw simulation) 
    status_text.text("Raw dispersion simulation...") 

    param_gaussian_model = ModelConfig(
        days=10,
        RH=RH,
        aerosol_type=NPS(nps),
        humidify=humidify,
        stability_profile=stability_type,
        stability_value=stability_value,
        wind_type=wind_type,
        wind_speed=wind_speed,
        output=OutputType.PLAN_VIEW,
        stacks=stacks,
        dry_size=dry_size, x_slice=26, y_slice=1,
        dispersion_model=DispersionModelType.PLUME)

    bounds = (payload["min_lon"], payload["min_lat"], payload["max_lon"], payload["max_lat"])

    response_gauss = requests.post(f"{API_URL}8002/start_simulation",
                                   json={"config": param_gaussian_model.to_dict(),
                                         "bounds": bounds})
        
    if response_gauss.status_code != 200:
        st.error("Error in Gaussian puff simulation.")
        return sensors_substance, substance_nps, None, None, None, metadata
        
    gauss_data = response_gauss.json()
    x = gauss_data.get("x", [])
    y = gauss_data.get("y", [])
    wind_dir = gauss_data.get("wind_dir")
    wind_speed = gauss_data.get("wind_speed")
    dispersion_map = gauss_data.get("predictions", [])

    status_text.text("Dispersion map generation...")
    plot_plan_view(dispersion_map, x, y, title="Raw Dispersion Map",
                   stability_class=stability_value.value)

    progress += 20
    progress_bar.progress(progress)

    # --- Dispersion simulation + correction
    status_text.text("Dispersion simulation...")
    response_mcxm = requests.post(f"{API_URL}8001/correct_dispersion", 
                                  json={ "wind_speed": wind_speed, 
                                        "wind_dir": wind_dir.tolist(),
                                        "concentration_map": dispersion_map.tolist(),
                                        "building_map": binary_map.tolist(), 
                                        "global_features": list(metadata.values()) }) 
    
    if response_mcxm.status_code != 200: 
        st.error("Errore nella correzione della dispersione.") 
        return sensors_substance, substance_nps, origin_lat, origin_lon, dispersion_map, metadata 
    
    real_dispersion_map = response_mcxm.json().get("predictions", [])
    
    progress += 20
    progress_bar.progress(progress)

    if map_section is not None:
        plot_dispersion_on_map(payload["min_lat"], payload["min_lon"], 
                                       payload["max_lat"], payload["max_lon"], 
                                       sensors_substance, dispersion_map, origin_lat, origin_lon)
        map_section.subheader("🗺️ Dispersion map")

    progress = 100
    progress_bar.progress(progress)
    status_text.text("Simulation completed ✅")

    st.session_state.simulation_results = {
        "weather": {"wind_speed": wind_speed, "wind_type": wind_type, "stability": stability_type, "RH": RH},
        "sensors": sensors_substance,
        "nps": substance_nps,
        "source": (origin_lat, origin_lon),
        "dispersion_map": real_dispersion_map,
        "metadata": metadata
    }

# ---------------- INTERFACCIA STREAMLIT ---------------- #
st.set_page_config(page_title="PentionSystem", layout="wide")
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = {
        "weather": None,
        "sensors": None,
        "nps": None,
        "source": None,
        "dispersion_map": None,
        "metadata": None
    }

st.markdown(
    """
    <div style="
        position: sticky; 
        top: 0; 
        background-color: white; 
        padding: 20px; 
        z-index: 999; 
        font-size: 36px; 
        font-weight: bold;
        text-align: center;
    ">
        💊 PENTION - NPS Source emission identification system
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar input
st.sidebar.header("Insert simulation parameters")
min_lat = st.sidebar.number_input("Min Lat", value=41.89, format="%.5f")
min_lon = st.sidebar.number_input("Min Lon", value=12.48, format="%.5f")
max_lat = st.sidebar.number_input("Max Lat", value=41.91, format="%.5f")
max_lon = st.sidebar.number_input("Max Lon", value=12.50, format="%.5f")
place = st.sidebar.text_input("Place", value="Insert place name")
n_sensors = st.sidebar.slider("Number of sensors", min_value=5, max_value=50, value=10, step=1)

st.sidebar.markdown(
    """
    <style>
    .start-btn > button {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
        padding: 0.5em 0;
    }
    .stop-btn > button {
        background-color: #dc3545 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
        padding: 0.5em 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.sidebar.columns(2)

with col1:
    st.markdown('<div class="start-btn">', unsafe_allow_html=True)
    start = st.button("▶ Start")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    stop = st.button("⏹ Stop")
    st.markdown('</div>', unsafe_allow_html=True)

# Layout colonne: lato-sinistra, centro (mappa), lato-destra
col_left, col_center, col_right = st.columns([1, 3, 1])

with col_left:
    weather_section = st.container()
    dispersion_section = st.container()

with col_center:
    map_section = st.container()

with col_right:
    nps_section = st.container()
    source_section = st.container()
    wind_rose_section = st.container()

with weather_section:
    st.markdown("**⛅Meteo conditions:**")
    weather_placeholder = st.empty()
    weather_placeholder.markdown(
        f"💨 **Wind speed (m/s):** N/A  \n"
        f"💨 **Wind type:** N/A  \n"
        f"📈 **Stability:** N/A  \n"
        f"♒︎ **Relative Humidity (%):** N/A"
    )

with dispersion_section:
    st.markdown("**🗺️ Dispersion map:**")
    dispersion_placeholder = st.empty()

sensors_section = st.container()

with sensors_section:
    st.markdown("**🛰️ Sensor:**")
    sensors_placeholder = st.empty()
    sensors_placeholder.write("No data available.")

with nps_section:
    st.markdown("**🧪 Nps predicted by sensor:**")
    nps_placeholder = st.empty()
    nps_placeholder.write("N/A")

with source_section:
    st.markdown("**📍 Source estimated:**")
    source_placeholder = st.empty()
    source_placeholder.write("N/A")

with wind_rose_section:
    st.markdown("🧭 **Wind rose**")
    wind_rose_placeholder = st.empty()

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

# ---------------- START SIMULATION ---------------- #
if start:

    status_text.success("Simulation started ✅")

    payload = {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "grid_size": 500,
        "place": place,
        "Number of sensors": n_sensors
    }

    run_application(payload)

elif stop:

    st.session_state.simulation_results = {
        "weather": None,
        "sensors": None,
        "nps": None,
        "source": None,
        "dispersion_map": None,
        "metadata": None
    }
    progress_bar.progress(0)
    status_text.text("Simulation stopped ❌")

    weather_placeholder.markdown(
        f"💨 **Wind speed (m/s):** N/A  \n"
        f"💨 **Wind type:** N/A  \n"
        f"📈 **Stability:** N/A  \n"
        f"♒︎ **Relative Humidity (%):** N/A"
    )
    sensors_placeholder.write("No data available.")
    nps_placeholder.write("N/A")
    source_placeholder.write("N/A")
    wind_rose_placeholder.empty()
    dispersion_placeholder.empty()
    map_section.empty()
else:
    results = st.session_state.simulation_results

    if results["weather"] is not None:
        weather_placeholder.markdown(
            f"- **Wind speed (m/s):** {results['weather']['wind_speed']}  \n"
            f"- **Wind type:** {results['weather']['wind_type']}  \n"
            f"- **Stability:** {results['weather']['stability']}  \n"
            f"- **Relative Humidity (%):** {results['weather']['RH']}"
        )

    if results["sensors"] is not None:
        sensor_info = [{"ID": s.id, "x": s.x, "y": s.y, "Status": "Operating" if not s.is_fault else "Faulty"}
                       for s in results["sensors"]]
        sensors_placeholder.table(sensor_info)

    if results["nps"] is not None:
        if results["nps"]:
            nps_placeholder.write(results["nps"])
        else:
            nps_placeholder.warning("No NPS identified.")

    if results["source"] is not None:
        origin_lat, origin_lon = results["source"]
        if origin_lat is not None and origin_lon is not None:
            source_placeholder.write(f"Lat: {origin_lat}, Long: {origin_lon}")
        else:
            source_placeholder.warning("Source not estimated.")