import streamlit as st
from plot_functions import *
from CorrectionDispersion.binary_map_gen import generate_binary_map
from controllers import *

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
    metadata_section = st.container()

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

with metadata_section:
    st.markdown("**🏙️ Info city map**")
    metadata_placeholder = st.empty()

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
        "place": place
    }

    run_application(payload,
        progress_bar, status_text,
                        weather_placeholder, sensors_placeholder, nps_placeholder,
                        source_placeholder, wind_rose_placeholder, map_section,
                        dispersion_placeholder, metadata_placeholder)

    # --- Binary map generation
    status_text.text("Binary map generation...")
    binary_map, metadata, free_cells, building_cells = generate_binary_map(payload)

    with metadata_section:
        metadata_placeholder.markdown(
            f"Griglia: {metadata.get('grid_size', 'N/A')}×{metadata.get('grid_size', 'N/A')}\n"
            f"Edifici totali: {metadata.get('total_buildings', 'N/A')}\n"
            f"Celle edifici: {int(np.sum(building_cells)) if isinstance(building_cells, np.ndarray) else building_cells:,}\n"
            f"Celle libere: {int(np.sum(free_cells)) if isinstance(free_cells, np.ndarray) else free_cells:,}\n"
            f"CRS: {metadata.get('crs', 'N/A')}\n"
            f"Risoluzione: {metadata.get('resolution (m)', 'N/A')} m\n"
            f"Densità edifici: {float(metadata.get('building_density', np.nan)):.1f}%\n"
            f"Altezza media edifici: {float(metadata.get('mean_height', np.nan))} m\n"
            f"Città: {metadata.get('city', 'N/A')}"
        )

    progress += 20
    progress_bar.progress(progress)

    # --- Meteo condition
    status_text.text("Sample meteo condition...")

    wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH = meteorological_condition()

    if weather_section is not None:
        weather_placeholder.markdown(
            f"💨 **Wind speed (m/s):** {wind_speed}  \n"
            f"💨 **Wind type:** {wind_type}  \n"
            f"📈 **Stability:** {stability_type}  \n"
            f"♒︎ **Relative Humidity (%):** {RH}"
        )

    # --- Sensor substance
    status_text.text("Air sampling...")
    sensors_substance, mass_spectrum = air_sample(n_sensors,free_cells, wind_speed, stability_type, RH, wind_type)

    plot_binary_map(binary_map, metadata['bounds'], map_section, sensors_substance)

    if sensors_section is not None:
        sensor_info = [{"ID": s.id, "x": s.x, "y": s.y, "Status": "Operating" if not s.is_fault else "Faulty",}
                       for s in sensors_substance]
        sensors_placeholder.table(sensor_info)

    progress += 20
    progress_bar.progress(progress)

    # --- NPS classification
    status_text.text("NPS classification...")

    prediction, nps = nps_classificator(mass_spectrum)

    if nps_section is not None:
        nps_placeholder.markdown(prediction)

    progress += 20
    progress_bar.progress(progress)

    # --- Localizzazione sorgente
    status_text.text("Source estimation...")
    origin_lat, origin_lon = estimate_location(sensors_substance)

    if source_section is not None:
        if origin_lat is not None and origin_lon is not None:
            source_placeholder.markdown(f"Lat: {origin_lat}, Long: {origin_lon}")
        else:
            source_placeholder.warning("Source not estimated.")

    progress += 20
    progress_bar.progress(progress)

    # --- gaussian plume dispersion (raw simulation)
    status_text.text("Raw dispersion simulation...")

    status_text.text("Dispersion map generation...")
    plot_plan_view(C1, x, y, dispersion_placeholder)
    status_text.text("Wind rose graph generation...")
    plot_wind_rose(wind_dir, wind_speed, wind_rose_placeholder)

    progress += 20
    progress_bar.progress(progress)

    # --- Dispersion simulation + correction
    status_text.text("Dispersion simulation...")

    _, _ = correction_dispersion()

    if map_section is not None:
        plot_dispersion_on_map(payload["min_lat"], payload["min_lon"],
                               payload["max_lat"], payload["max_lon"],
                               sensors_substance, C1, origin_lat, origin_lon)
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