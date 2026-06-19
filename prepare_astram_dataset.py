import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import json

# Constants
try:
    with open("data/city_network.json", "r") as f:
        network_data = json.load(f)
        CORRIDORS = network_data["CORRIDORS"]
        COORDS = network_data["coords"]
except FileNotFoundError:
    print("Warning: data/city_network.json not found. Falling back to default empty structures.")
    CORRIDORS = []
    COORDS = {}

N = len(CORRIDORS)
CORRIDOR_TO_IDX = {c: i for i, c in enumerate(CORRIDORS)}


class TrafficDataPipeline:
    def __init__(self, test_mode=False, output_dir="dataset/AstramBengaluru"):
        self.test_mode = test_mode
        self.output_dir = output_dir
        
        self.start_date = datetime(2023, 11, 9, 0, 0, 0)
        self.days = 10 if test_mode else 365
        self.steps_per_day = 144
        self.T = self.days * self.steps_per_day
        self.time_indices = [self.start_date + timedelta(minutes=10 * t) for t in range(self.T)]
        
        self.A_static = None
        self.V_base = None
        self.L_event = None
        self.eod_counts = None
        self.A_dynamic = None
        self.V_final = None

    def build_static_graph(self):
        print("Building static graph topology using Gaussian Distance Kernel...")
        self.A_static = np.zeros((N, N), dtype=np.float32)
        
        # Calculate pairwise geographical distances
        dist_matrix = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            lat1, lon1 = COORDS[CORRIDORS[i]]
            for j in range(N):
                lat2, lon2 = COORDS[CORRIDORS[j]]
                # simple Euclidean distance of lat/lon
                dist = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
                dist_matrix[i, j] = dist
                
        # Calculate standard deviation of distances for the kernel
        distances = dist_matrix[~np.eye(dist_matrix.shape[0], dtype=bool)]
        std_dist = distances.std()
        
        # Apply Gaussian Kernel: W_ij = exp(- (dist_ij^2 / std_dist^2))
        for i in range(N):
            for j in range(N):
                if i != j:
                    weight = np.exp(- (dist_matrix[i, j] ** 2) / (std_dist ** 2))
                    # Thresholding to create a sparse graph
                    if weight >= 0.1:
                        self.A_static[i, j] = weight
        return self.A_static

    def generate_synthetic_baseline(self):
        print("Simulating baseline speed profiles...")
        V_free = 50.0
        self.V_base = np.zeros((self.T, N))
        np.random.seed(42)
        
        for t in range(self.T):
            t_day = t % self.steps_per_day
            dow = (t // self.steps_per_day) % 7
            is_weekend = dow in [5, 6]
            weekend_factor = 0.5 if is_weekend else 1.0
            
            for n in range(N):
                phase_shift = (n * 2) % 6
                t_shifted = t_day + phase_shift
                p1 = np.exp(-((t_shifted - 57) / 12) ** 2)
                p2 = np.exp(-((t_shifted - 111) / 15) ** 2)
                
                v = V_free - (15.0 * p1 + 22.0 * p2) * weekend_factor
                v += 2.0 * np.sin(2 * np.pi * t / (self.steps_per_day * 7))
                v += np.random.normal(0, 1.2)
                self.V_base[t, n] = np.clip(v, 12.0, 60.0)
        return self.V_base

    def load_historical_baseline(self, traffic_csv):
        print(f"Loading raw historical traffic from {traffic_csv}...")
        # Placeholder for future actual traffic data ingestion
        # Format expected: timestamp, corridor, speed
        raise NotImplementedError("Historical traffic CSV format not yet defined. Falling back to synthetic.")

    def process_events(self, csv_path):
        print("Reading and mapping Astram events...")
        df = pd.read_csv(csv_path)
        df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
        df['end_datetime'] = pd.to_datetime(df['end_datetime'], errors='coerce')

        self.L_event = np.zeros((self.T, N))
        self.eod_counts = np.zeros((self.T, N))
        active_events_list = []

        print("Mapping events to timelines and calculating structural shocks...")
        for idx, row in df.iterrows():
            c_name = row['corridor']
            if pd.isna(c_name) or c_name == 'nan':
                lat, lon = row['latitude'], row['longitude']
                if pd.isna(lat) or pd.isna(lon):
                    c_name = 'Non-corridor'
                else:
                    best_c = 'Non-corridor'
                    best_d = float('inf')
                    for c_cand, coord in COORDS.items():
                        if c_cand == 'Non-corridor': continue
                        d = np.sqrt((lat - coord[0])**2 + (lon - coord[1])**2)
                        if d < best_d:
                            best_d = d
                            best_c = c_cand
                    c_name = best_c

            if c_name not in CORRIDOR_TO_IDX:
                c_name = 'Non-corridor'

            c_idx = CORRIDOR_TO_IDX[c_name]
            
            start_dt = row['start_datetime']
            if pd.isna(start_dt):
                continue
                
            end_dt = row['end_datetime']
            priority = str(row['priority']).lower()
            
            if 'high' in priority: s_base = 1.0
            elif 'medium' in priority: s_base = 0.6
            else: s_base = 0.3
                
            if bool(row.get('requires_road_closure', False)):
                s_base = min(1.0, s_base + 0.3)
                
            if pd.isna(end_dt):
                dur_hours = 6 if s_base >= 0.8 else (4 if s_base >= 0.5 else 2)
                end_dt = start_dt + timedelta(hours=dur_hours)
                
            t_start_idx = int((start_dt.tz_localize(None) - self.start_date).total_seconds() // 600)
            t_end_idx = int((end_dt.tz_localize(None) - self.start_date).total_seconds() // 600)
            
            t_start_idx = max(0, min(self.T - 1, t_start_idx))
            t_end_idx = max(0, min(self.T - 1, t_end_idx))
            
            if t_end_idx < t_start_idx:
                t_end_idx, t_start_idx = t_start_idx, t_end_idx
            if t_start_idx == t_end_idx:
                t_end_idx += 1
                
            active_events_list.append((t_start_idx, t_end_idx, c_idx, s_base))
            
            for t_step in range(t_start_idx, min(self.T, t_end_idx + 12)):
                delta_t = t_step - t_start_idx
                decay = np.exp(-0.15 * delta_t)
                drag = 25.0 * s_base * decay
                if t_step < self.T:
                    self.L_event[t_step, c_idx] = max(self.L_event[t_step, c_idx], drag)
                    self.eod_counts[t_step, c_idx] = min(3.0, self.eod_counts[t_step, c_idx] + s_base)

        print("Applying event degradation shock to baseline velocities...")
        self.V_final = np.maximum(5.0, self.V_base - self.L_event)

        print("Generating dynamic, event-conditioned network topologies (Dynamic Graph Orbit)...")
        self.A_dynamic = np.repeat(self.A_static[np.newaxis, :, :], self.T, axis=0)

        for t_start, t_end, c_idx, severity in active_events_list:
            for t_step in range(t_start, t_end):
                if t_step >= self.T: break
                factor_self = 1.0 - 0.7 * severity
                self.A_dynamic[t_step, c_idx, :] *= factor_self
                self.A_dynamic[t_step, :, c_idx] *= factor_self
                
                factor_neigh = 1.0 - 0.3 * severity
                neighbors = np.where(self.A_static[c_idx] > 0)[0]
                for n_idx in neighbors:
                    self.A_dynamic[t_step, n_idx, :] *= factor_neigh
                    self.A_dynamic[t_step, :, n_idx] *= factor_neigh

    def export_features(self):
        print("Normalizing features and exporting files...")
        mean_speed = self.V_final.mean(axis=0)
        std_speed = self.V_final.std(axis=0)
        std_speed[std_speed == 0] = 1.0
        
        norm_var = (self.V_final - mean_speed) / std_speed
        
        norm_time_marker = np.zeros((self.T, 5))
        for t in range(self.T):
            t_day = t % self.steps_per_day
            dow = (t // self.steps_per_day) % 7
            dom = ((t // self.steps_per_day) % 30)
            doy = ((t // self.steps_per_day) % 365)
            
            norm_time_marker[t, 0] = t_day / (self.steps_per_day - 1)
            norm_time_marker[t, 1] = dow / 6.0
            norm_time_marker[t, 2] = dom / 29.0
            norm_time_marker[t, 3] = doy / 364.0
            norm_time_marker[t, 4] = np.mean(self.eod_counts[t]) / 3.0

        os.makedirs(self.output_dir, exist_ok=True)

        np.savez(os.path.join(self.output_dir, "feature.npz"), norm_var=norm_var, norm_time_marker=norm_time_marker)
        np.savez(os.path.join(self.output_dir, "var_scaler_info.npz"), mean=mean_speed, std=std_speed)
        np.save(os.path.join(self.output_dir, "adj_mat.npy"), self.A_static)
        np.save(os.path.join(self.output_dir, "adj_mat_dynamic.npy"), self.A_dynamic)

        coords_df = pd.DataFrame([{"corridor": c, "latitude": COORDS[c][0], "longitude": COORDS[c][1]} for c in CORRIDORS])
        coords_df.to_csv(os.path.join(self.output_dir, "corridor_coordinates.csv"), index=False)

        print("Pipeline features generated successfully!")
        print(f"norm_var shape: {norm_var.shape}")
        print(f"norm_time_marker shape: {norm_time_marker.shape}")
        print(f"adj_mat shape: {self.A_static.shape}")
        print(f"adj_mat_dynamic shape: {self.A_dynamic.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode with short dataset")
    parser.add_argument("--traffic_csv", type=str, default=None, help="Optional: Path to raw historical traffic data")
    parser.add_argument("--event_csv", type=str, default=r"dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv", help="Path to Astram event dataset CSV")
    args = parser.parse_args()

    print("--- Starting prepare_astram_dataset.py ---")
    event_csv_path = args.event_csv
    if not os.path.exists(event_csv_path):
        raise FileNotFoundError(f"Astram event dataset CSV not found at: {event_csv_path}")

    pipeline = TrafficDataPipeline(test_mode=args.test)
    pipeline.build_static_graph()
    
    if args.traffic_csv and os.path.exists(args.traffic_csv):
        pipeline.load_historical_baseline(args.traffic_csv)
    else:
        pipeline.generate_synthetic_baseline()
        
    pipeline.process_events(event_csv_path)
    pipeline.export_features()
    print("--- prepare_astram_dataset.py execution completed ---")

if __name__ == "__main__":
    main()
