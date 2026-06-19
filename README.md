# 🛰️ Flipkart Gridlock 2.0: Event-Driven Congestion

A fully operational, AI-driven traffic decision support system built for Bengaluru's complex urban road networks. This system acts as an end-to-end "Mission Control" center. It forecasts gridlock shockwaves from unplanned events, recommends optimal diversion routes, mathematically allocates police manpower, and generates tactical physical infrastructure strategies (barricading, traffic signals, and active CCTV monitoring).

## 🏆 Key Capabilities

### 1. Hybrid AI Forecasting & External Data Engine
- Integrates Deep Learning models trained on historical Astram traffic data.
- **External Data Integration:** Expanding standard temporal features to an 8-dimensional tensor that actively parses weather (rain), public holidays, and school schedules to adjust congestion spread predictions dynamically.
- Employs physics-based exponential decay shockwaves to dynamically simulate the cascading spillover of localized accidents or planned events.

### 2. Live Unplanned Event Auto-Mitigation
- Connects to live traffic streams to instantly detect anomalous congestion drops using the `AnomalyDetector`.
- Zero-touch operation: Automatically injects detected anomalies into the Mission Control UI, instantly generating mitigation strategies and rerouting options without human intervention.

### 3. Time-Dependent Routing Engine
- Computes **K-Shortest Diversion Paths** in real-time, accounting for future traffic deterioration rather than just current static conditions.
- Uses dynamic edge-penalization to provide **Plan A, Plan B, and Plan C** routes, ensuring diverted traffic doesn't create secondary gridlocks.

### 4. Mission Control Tactical Dashboards
- **👮 Manpower Optimizer**: A greedy search algorithm that takes a constrained budget of police officers and outputs the mathematically optimal deployment locations across the impact zone to maximize network speed, accounting for driving distance from Police HQ.
- **🚧 Infrastructure Strategy**: Automatically maps feeder corridors and diversion routes to recommend exact locations for barricade placement and +30s Green Phase signal extensions.
- **📹 Intersection Monitoring Alerts**: Dynamically flags high-risk, unmitigated neighbor intersections experiencing spillover shockwaves to the control room for active CCTV monitoring.
- **📏 Queue Length Estimation**: Bridges the gap between abstract AI speeds and physical reality by calculating real-time vehicle queues in explicit kilometers and trapped vehicle counts.

### 5. Automated Post-Event Learning (Retraining Loop)
- Features a "Post-Event Analysis" module allowing the system to seamlessly evaluate ground-truth metrics against AI predictions.
- **One-Click Retraining:** Appends validated scenarios directly into the `feature.npz` dataset and triggers an asynchronous background Ray Tune/PyTorch Lightning retraining job to ensure the AI constantly improves.

---

## 📂 Project Structure

### Core Components
- **`dashboard.py`**: The main entry point for the Streamlit application providing the "Mission Control" UI.
- **`recommendation_engine.py`**: The algorithmic heart of the platform handling `TDSPGraph`, `PolicySimulator`, `ManpowerOptimizer`, and `InfrastructureOptimizer`.
- **`train.py`**: PyTorch Lightning training script for the spatiotemporal forecasting models.

### Scripts (`script/`)
- **`inference_api.py`**: Loads the trained PyTorch checkpoint and executes true deterministic tensor forward passes to predict speed drops based on severity and external weather/holiday inputs.
- **`anomaly_detector.py`**: Unplanned event auto-mitigation logic.
- **`traffic_metrics.py`**: Generates physics-based congestion severity, delay times, and physical queue lengths.
- **`live_traffic_api.py`**: Connects to the live API or simulates a mock historical database stream.
- **`append_retrain.py`**: The script responsible for appending validated event matrices to the dataset and launching background subprocess retraining.

---

## 🚀 Detailed Setup Guide

### Prerequisites
- **Python 3.10 to 3.12**
- Git
- Recommended: Windows / Ubuntu with CUDA-capable GPU for faster AI training.

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rajat552/Event-Driven-Congestion.git
   cd Event-Driven-Congestion
   ```

2. **Create a Virtual Environment**
   It is highly recommended to isolate dependencies.
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Install the required data science, mapping, and AI libraries.
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate the Core Dataset & Synthesize External Features**
   Before running the AI, you must build the dynamic adjacency matrices and the 8-dimensional temporal feature tensor (which synthesizes rain, holidays, and school schedules across 365 days).
   ```bash
   python prepare_astram_dataset.py
   ```
   *Note: To test quickly with a shorter 10-day dataset, you can run `python prepare_astram_dataset.py --test`.*

5. **Train the AI Model**
   Run the PyTorch Lightning trainer to establish your baseline checkpoint. The configuration handles spatial-temporal deep learning optimization automatically.
   ```bash
   python train.py -c config/AstramBengaluru/STIDEF.py
   ```

6. **Launch the Mission Control Dashboard**
   Boot up the Streamlit UI. The dashboard automatically discovers the most recently trained `.ckpt` weights in the `save/` directory.
   ```bash
   streamlit run dashboard.py
   ```

### ⚙️ Environment Variables (Optional)
The system supports decoupling hardcoded paths. You can customize the environment by setting the following variables before running Streamlit:
- `CITY_ID`: Override the dataset directory (Default: `AstramBengaluru`).
- `MODEL_CHECKPOINT_DIR`: Override the search path for AI `.ckpt` weights.

## 🛠️ Usage Example

1. Open `http://localhost:8501`.
2. Expand **"Simulate An Anomaly Event"** on the sidebar.
3. Select an origin corridor (e.g., *Hosur Road*), designate it as a "High Severity Accident" causing a road closure.
4. Click **Add Event**.
5. Observe the Mission Control dashboard instantly execute a real-time Time-Dependent Shortest Path route.
6. Slide the **Deploy Police Personnel** and **Deploy Barricades** sliders in the sidebar to dynamically witness the AI forecast adjusting its physical queue length estimates and updating the active CCTV monitoring nodes in response to your resource allocations!
