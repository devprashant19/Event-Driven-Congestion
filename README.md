# 🛰️ Flipkart Gridlock 2.0: Event-Driven Congestion

A fully operational, AI-driven traffic decision support system built for Bengaluru's complex urban road networks. This system goes beyond static traffic forecasting by acting as an end-to-end "Mission Control" center. It forecasts gridlock shockwaves from unplanned events, recommends optimal diversion routes, allocates police manpower mathematically, and generates tactical physical infrastructure strategies (barricading & traffic signals).

## 🏆 Key Features

### 1. Hybrid AI Forecasting Engine
- Integrates State-of-the-Art Deep Learning models (**T3STID / T3GWNet**) trained on historical Astram traffic data.
- Employs physics-based exponential decay shockwaves to dynamically simulate the cascading spillover of localized accidents or planned events.

### 2. Time-Dependent Routing Engine
- Computes **K-Shortest Diversion Paths** in real-time, accounting for future traffic deterioration rather than just current static conditions.
- Uses dynamic edge-penalization to provide **Plan A, Plan B, and Plan C** routes, ensuring diverted traffic doesn't create secondary gridlocks.

### 3. Mission Control Tactical Dashboards
- **👮 Manpower Optimizer**: A greedy search algorithm that takes a constrained budget of police officers and outputs the mathematically optimal deployment locations across the impact zone to maximize network speed.
- **🚧 Infrastructure Strategy**: Automatically maps feeder corridors and diversion routes to recommend exact locations for barricade placement and +30s Green Phase signal extensions.
- **📊 Post-Event Learning Loop**: A dedicated analysis tab to compare the AI's Unmitigated Baseline predictions against Actual Ground Truth, providing an Intervention Success Score for continuous retraining.
- **📅 Interactive What-If Scenarios**: Build complex operational plans by scheduling multiple future events on a timeline and visualizing their intersecting gridlock zones.

## 🚀 How to Run

### Option 1: Native Python (Virtual Environment)
```bash
# 1. Activate the environment
source venv/bin/activate

# 2. Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. Launch the Mission Control Dashboard
streamlit run dashboard.py
```

### Option 2: Docker Deployment
```bash
# 1. Build the image
docker build -t gridlock2 .

# 2. Run the container
docker run -p 8501:8501 gridlock2
```
Access the dashboard at `http://localhost:8501`

## 📂 Architecture Overview
- `dashboard.py`: The main Streamlit Mission Control interface with 3D PyDeck maps.
- `recommendation_engine.py`: The brain behind the routing algorithms, greedy manpower optimizer, and infrastructure barricade mapping.
- `script/inference_api.py`: Wraps the PyTorch Lightning STGNN models for real-time predictions.
- `script/anomaly_detector.py`: Detects real-time deviations from expected traffic norms and calculates impact radii.
- `easytsf/`: The core model architecture library containing STIDEF and other Spatial-Temporal networks.
