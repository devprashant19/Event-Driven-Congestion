import os
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from recommendation_engine import TDSPGraph, PolicySimulator, CORRIDORS, CORRIDOR_LENGTHS, coords, N, corridor_to_idx


# Page Config
st.set_page_config(
    page_title="Flipkart Gridlock 2.0 - Event-Driven Congestion",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Zero-Gravity CSS Theme Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 700;
    }
    .delta-positive {
        color: #10b981 !important;
        font-weight: 600;
    }
    .delta-negative {
        color: #ef4444 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- SESSION STATE & INITIALIZATION -----------------
if 'simulated_event' not in st.session_state:
    st.session_state.simulated_event = None

if 'simulated_speeds' not in st.session_state:
    st.session_state.simulated_speeds = None

if 'simulated_adj' not in st.session_state:
    st.session_state.simulated_adj = None

# Paths to data
static_adj_path = r"dataset/AstramBengaluru/adj_mat.npy"
scaler_path = r"dataset/AstramBengaluru/var_scaler_info.npz"

# Instantiate simulators
@st.cache_resource
def load_policy_simulator():
    if os.path.exists(static_adj_path) and os.path.exists(scaler_path):
        return PolicySimulator(static_adj_path, scaler_path)
    return None

sim = load_policy_simulator()
A_static = np.load(static_adj_path) if os.path.exists(static_adj_path) else None

# Handle fallbacks if dataset isn't fully generated
if sim is None or A_static is None:
    st.error("Error: Adjacency matrix or scaler info not found. Please ensure the dataset pipeline is executed.")
    st.stop()

# ----------------- TITLE / HEADER -----------------
st.title("🛰️ Flipkart Gridlock 2.0: Event-Driven Congestion")
st.markdown("### Zero-Gravity Mission Control Traffic Decision Support System")
st.write("Refactoring spatial-temporal traffic pipelines with dynamic graph topologies, time-dependent routing, and continuous synthetic baselines.")

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("🛠️ Policy Simulator Controls")

# Sliders for police and barricades
officers = st.sidebar.slider("👮 Deploy Police Personnel", min_value=0, max_value=50, value=10, step=1)
barricades = st.sidebar.slider("🚧 Deploy Barricades", min_value=0, max_value=20, value=4, step=1)

st.sidebar.markdown("---")
st.sidebar.header("🚨 Simulate An Anomaly Event")

sim_corridor = st.sidebar.selectbox("Select Target Corridor", CORRIDORS)
sim_cause = st.sidebar.selectbox("Event Cause", ["Political Rally", "Festival", "Sports Match", "Construction", "Vehicle Breakdown", "Accident"])
sim_severity = st.sidebar.select_slider("Incident Severity", options=["Low", "Medium", "High"])
road_closure = st.sidebar.checkbox("Requires Full Road Closure")

# Map text severity to numeric
severity_map = {"Low": 0.3, "Medium": 0.6, "High": 1.0}

if st.sidebar.button("⚡ Inject Incident into Graph"):
    severity_val = severity_map[sim_severity]
    if road_closure:
        severity_val = min(1.0, severity_val + 0.3)
    
    # Simulate an incident spanning from timestep 4 to step 16 (approx 2 hours)
    st.session_state.simulated_event = {
        "corridor": sim_corridor,
        "c_idx": corridor_to_idx[sim_corridor],
        "start": 4,
        "end": 16,
        "severity": severity_val,
        "cause": sim_cause
    }
    st.sidebar.success(f"Incident injected successfully on {sim_corridor}!")

if st.sidebar.button("🔄 Reset Graph Topology"):
    st.session_state.simulated_event = None
    st.session_state.simulated_speeds = None
    st.session_state.simulated_adj = None
    st.sidebar.info("Graph reset to static baseline.")

# ----------------- DYNAMIC DATA COMPUTATION -----------------
# 1. Base sinusoidal velocities (24 steps for 4 hours simulation window)
steps_window = 24
np.random.seed(42)
base_speeds = np.zeros((steps_window, len(CORRIDORS)))
for t in range(steps_window):
    # Sinusoidal baseline drops (rush hour simulation)
    peak = np.exp(-((t - 10) / 6) ** 2) # peak around step 10
    for n in range(len(CORRIDORS)):
        v = 50.0 - 18.0 * peak + np.random.normal(0, 0.8)
        base_speeds[t, n] = np.clip(v, 15.0, 60.0)

# Apply policy simulation
if st.session_state.simulated_event is not None:
    evt = st.session_state.simulated_event
    event_list = [(evt["start"], evt["end"], evt["c_idx"], evt["severity"])]
    
    # 1. Unmitigated Speed (No interventions)
    speeds_unmit, adj_unmit = sim.simulate_mitigated_speed(base_speeds, event_list, 0, 0)
    
    # 2. Mitigated Speed (With current sliders values)
    speeds_mit, adj_mit = sim.simulate_mitigated_speed(base_speeds, event_list, officers, barricades)
else:
    speeds_unmit = base_speeds.copy()
    speeds_mit = base_speeds.copy()
    adj_mit = np.repeat(A_static[np.newaxis, :, :], steps_window, axis=0)

# Time Step Slider
selected_t = st.slider("🕰️ Time Progress (10-minute intervals)", min_value=0, max_value=steps_window - 1, value=6, step=1)

# Current state parameters
current_speeds = speeds_mit[selected_t]

# ----------------- METRIC CARDS ROW -----------------
col1, col2, col3, col4 = st.columns(4)

# Calculate Delays for OD Pairs to compute Delta
od_pairs = [
    (0, 10),  # Tumkur Road to Hosur Road
    (15, 1),  # Mysore Road to ORR East 1
    (17, 8),  # CBD 1 to Bellary Road 2
    (7, 11),  # Old Madras Road to Bannerghata Road
    (13, 4)   # Magadi Road to ORR East 2
]

graph_unmit = TDSPGraph(A_static, CORRIDOR_LENGTHS)
graph_mit = TDSPGraph(A_static, CORRIDOR_LENGTHS)

total_time_unmit = 0.0
total_time_mit = 0.0

for s_node, t_node in od_pairs:
    _, time_un = graph_unmit.solve_tdsp(s_node, t_node, selected_t * 10.0, speeds_unmit)
    _, time_mit = graph_mit.solve_tdsp(s_node, t_node, selected_t * 10.0, speeds_mit)
    total_time_unmit += time_un
    total_time_mit += time_mit

delta_delay = max(0.0, total_time_unmit - total_time_mit)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Network Speed</div>
        <div class="metric-value">{current_speeds.mean():.1f} km/h</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Unmitigated Cumulative Delay</div>
        <div class="metric-value">{total_time_unmit:.1f} mins</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Mitigated Cumulative Delay</div>
        <div class="metric-value">{total_time_mit:.1f} mins</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    delta_class = "delta-positive" if delta_delay > 0 else "delta-negative"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Δ (Delta) Delay Reduced</div>
        <div class="metric-value {delta_class}">-{delta_delay:.1f} mins</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- GEOSPATIAL MAP VIEW & LAYOUT -----------------
map_col, route_col = st.columns([2, 1])

with map_col:
    st.subheader("🗺️ Dynamic Network Topology Map")
    
    # Build dynamic nodes DataFrame for Pydeck
    nodes_data = []
    for i, name in enumerate(CORRIDORS):
        lat, lon = coords[name]
        speed = current_speeds[i]
        
        # Color coding
        if speed >= 40:
            color = [0, 220, 100, 200]  # Green
        elif speed >= 25:
            color = [255, 160, 0, 200]  # Orange
        else:
            color = [255, 50, 50, 220]  # Red
            
        radius = 450
        if st.session_state.simulated_event is not None:
            if i == st.session_state.simulated_event["c_idx"]:
                radius = 800  # Expand radius to represent the incident node
                
        nodes_data.append({
            "name": name,
            "coordinates": [lon, lat],
            "speed": speed,
            "color": color,
            "radius": radius
        })
    nodes_df = pd.DataFrame(nodes_data)

    # Build connectivity lines for Pydeck
    lines_data = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_static[i, j] > 0:
                # Dynamic weight/capacity drop based on connected node speeds
                avg_speed = (current_speeds[i] + current_speeds[j]) / 2.0
                if avg_speed >= 40:
                    color = [0, 220, 100, 100]
                elif avg_speed >= 25:
                    color = [255, 160, 0, 100]
                else:
                    color = [255, 50, 50, 180]
                    
                lines_data.append({
                    "start": [coords[CORRIDORS[i]][1], coords[CORRIDORS[i]][0]],
                    "end": [coords[CORRIDORS[j]][1], coords[CORRIDORS[j]][0]],
                    "color": color
                })
    lines_df = pd.DataFrame(lines_data)

    # Pydeck Layers
    line_layer = pdk.Layer(
        "LineLayer",
        data=lines_df,
        get_source_position="start",
        get_target_position="end",
        get_color="color",
        get_width=4,
        pickable=False
    )

    scatterplot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nodes_df,
        get_position="coordinates",
        get_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True
    )

    view_state = pdk.ViewState(
        latitude=12.9716,
        longitude=77.5946,
        zoom=11.2,
        pitch=25
    )

    deck = pdk.Deck(
        layers=[line_layer, scatterplot_layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}\nPredicted Speed: {speed:.1f} km/h"}
    )
    st.pydeck_chart(deck)

with route_col:
    st.subheader("🧭 Time-Dependent Router")
    st.write("Calculate shortest paths dynamically accounting for future traffic congestion.")

    origin = st.selectbox("Origin Corridor", CORRIDORS, index=0)
    destination = st.selectbox("Destination Corridor", CORRIDORS, index=10)

    start_idx = corridor_to_idx[origin]
    target_idx = corridor_to_idx[destination]

    # Calculate routes
    path_unmit, time_unmit = graph_unmit.solve_tdsp(start_idx, target_idx, selected_t * 10.0, speeds_unmit)
    path_mit, time_mit = graph_mit.solve_tdsp(start_idx, target_idx, selected_t * 10.0, speeds_mit)

    st.markdown("#### Dynamic Routing Results:")
    if path_mit:
        st.markdown(f"**Route**: ` ➔ `.join([CORRIDORS[n] for n in path_mit])")
        st.markdown(f"⏱️ **Traversing Duration**: `{time_mit:.1f} minutes`")
        
        # Policy impact evaluation card
        diff = time_unmit - time_mit
        if diff > 0.5:
            st.success(f"🎉 Intervention saves **{diff:.1f} minutes** on this path!")
        else:
            st.info("Alternative route clear under current deployment.")
    else:
        st.error("No path found between selected nodes.")

    # Status Board of Injected Incident
    st.markdown("---")
    st.markdown("#### Event Status Board")
    if st.session_state.simulated_event is not None:
        evt = st.session_state.simulated_event
        st.error(f"⚠️ **ACTIVE EVENT**: {evt['cause']}")
        st.markdown(f"- **Location**: {evt['corridor']}")
        st.markdown(f"- **Severity Level**: {sim_severity}")
        st.markdown(f"- **Timeline Steps**: step {evt['start']} to {evt['end']}")
    else:
        st.success("🟢 **SYSTEM NORMAL**: No active events.")
