# 🛰️ Flipkart Gridlock 2.0: Event-Driven Congestion

A fully operational, AI-driven traffic decision support system built for Bengaluru's complex urban road networks. This system acts as an end-to-end "Mission Control" center. It forecasts gridlock shockwaves from unplanned events, recommends optimal diversion routes, mathematically allocates police manpower, and generates tactical physical infrastructure strategies (barricading & traffic signals).

## 🏆 Key Features

### 1. Hybrid AI Forecasting Engine
- Integrates Deep Learning models trained on historical Astram traffic data.
- Employs physics-based exponential decay shockwaves to dynamically simulate the cascading spillover of localized accidents or planned events.

### 2. Time-Dependent Routing Engine
- Computes **K-Shortest Diversion Paths** in real-time, accounting for future traffic deterioration rather than just current static conditions.
- Uses dynamic edge-penalization to provide **Plan A, Plan B, and Plan C** routes, ensuring diverted traffic doesn't create secondary gridlocks.

### 3. Mission Control Tactical Dashboards
- **👮 Manpower Optimizer**: A greedy search algorithm that takes a constrained budget of police officers and outputs the mathematically optimal deployment locations across the impact zone to maximize network speed.
- **🚧 Infrastructure Strategy**: Automatically maps feeder corridors and diversion routes to recommend exact locations for barricade placement and +30s Green Phase signal extensions.
- **📊 Post-Event Learning Loop**: A dedicated analysis tab to compare the AI's Unmitigated Baseline predictions against Actual Ground Truth, providing an Intervention Success Score for continuous retraining.

## 📂 Project Structure & Code Explanation

### Core Components
- **`dashboard.py`**: The main entry point for the Streamlit application. It provides the "Mission Control" UI, allowing users to inject simulated anomaly events, tweak policy variables (deploying police/barricades), and visualize dynamic map routes and bottlenecks via `pydeck`.
- **`recommendation_engine.py`**: The algorithmic heart of the platform. It includes:
  - `TDSPGraph`: Implements Time-Dependent Shortest Path routing to avoid areas that *will* be congested in the future.
  - `PolicySimulator`: Alters traffic flow dynamically based on the number of deployed officers and barricades.
  - `ManpowerOptimizer`: Uses a greedy allocation algorithm to distribute a limited number of police officers to intersections maximizing overall network speed.
  - `InfrastructureOptimizer`: Determines optimal placement for barricades to choke off flow into gridlocked areas and extends green phases along diversion routes.

### AI Training & Data Pipeline
- **`prepare_astram_dataset.py`**: Processes the raw ASTRAM event data, simulating baseline speed profiles, mapping events to network corridors, and exporting the topological/dynamic adjacency matrices and speed arrays used for training.
- **`train.py`**: PyTorch Lightning training script for the spatiotemporal forecasting models. It seamlessly handles configuration merging, Ray Tune integration for hyperparameter tuning, and metric logging.
- **`ray_tune.py`**: Facilitates hyperparameter search and model optimization via the Ray framework.

### Scripts (`script/`)
Contains support modules:
- **`inference_api.py`**: Loads the trained checkpoint and provides live inference predictions.
- **`anomaly_detector.py`**: Detects traffic anomalies by comparing predictions to threshold severity.
- **`traffic_metrics.py`**: Analyzes traffic severity and delays across the network.
- **`mock_stream.py`**: Simulates a live stream of incoming traffic data for testing the dashboard.

## 🚀 Getting Started

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Generate Dataset**: `python prepare_astram_dataset.py`
3. **Train Model**: `python train.py -c config/your_config.json`
4. **Launch Dashboard**: `streamlit run dashboard.py`
