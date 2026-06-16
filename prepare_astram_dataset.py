import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode with short dataset")
    args = parser.parse_args()

    print("--- Starting prepare_astram_dataset.py ---")
    csv_path = r"dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Astram dataset CSV not found at: {csv_path}")

    # 1. Define the 22 Bengaluru Corridors and their median coordinates
    CORRIDORS = [
        'Tumkur Road', 'ORR East 1', 'Non-corridor', 'CBD 2', 'ORR East 2',
        'ORR West 1', 'ORR North 1', 'Old Madras Road', 'Bellary Road 2',
        'Bellary Road 1', 'Hosur Road', 'Bannerghata Road', 'ORR North 2',
        'Magadi Road', 'IRR(Thanisandra road)', 'Mysore Road', 'West of Chord Road',
        'CBD 1', 'Old Airport Road', 'Hennur Main Road', 'Airport New South Road',
        'Varthur Road'
    ]
    N = len(CORRIDORS)
    print(f"Loaded {N} corridors.")

    corridor_to_idx = {c: i for i, c in enumerate(CORRIDORS)}
    
    # Coordinate mapping (calculated from average of event coordinates)
    coords = {
        'Tumkur Road': (13.03146, 77.53366),
        'ORR East 1': (12.92831, 77.66913),
        'Non-corridor': (12.98286, 77.59869),
        'CBD 2': (12.98331, 77.59505),
        'ORR East 2': (12.97583, 77.69603),
        'ORR West 1': (12.92084, 77.55913),
        'ORR North 1': (13.02455, 77.63744),
        'Old Madras Road': (12.98091, 77.62932),
        'Bellary Road 2': (13.10596, 77.60327),
        'Bellary Road 1': (13.01680, 77.58640),
        'Hosur Road': (12.91547, 77.62466),
        'Bannerghata Road': (12.89638, 77.59788),
        'ORR North 2': (13.04193, 77.55882),
        'Magadi Road': (12.98506, 77.52334),
        'IRR(Thanisandra road)': (12.93751, 77.62694),
        'Mysore Road': (12.95779, 77.56365),
        'West of Chord Road': (12.98297, 77.54634),
        'CBD 1': (12.98102, 77.60682),
        'Old Airport Road': (12.95887, 77.66185),
        'Hennur Main Road': (13.05115, 77.62619),
        'Airport New South Road': (13.02752, 77.63353),
        'Varthur Road': (12.95655, 77.71594),
    }

    # 2. Build the Static Adjacency Matrix A_static (N x N)
    # Using distances between corridor center coordinates, connect each to K=3 nearest neighbors
    A_static = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        distances = []
        lat1, lon1 = coords[CORRIDORS[i]]
        for j in range(N):
            if i == j:
                continue
            lat2, lon2 = coords[CORRIDORS[j]]
            dist = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
            distances.append((dist, j))
        
        # Sort and select nearest 3
        distances.sort()
        for _, j in distances[:3]:
            A_static[i, j] = 1.0
            A_static[j, i] = 1.0 # Undirected graph model

    # 3. Setup time grid (10-minute intervals)
    # Start: 2023-11-09 00:00:00, End: 2024-11-08 23:50:00 (365 days)
    start_date = datetime(2023, 11, 9, 0, 0, 0)
    if args.test:
        days = 10
        print("Running in TEST mode: limiting dataset to 10 days.")
    else:
        days = 365
        print("Running in PRODUCTION mode: generating 365 days dataset.")

    steps_per_day = 144  # 24 * 6
    T = days * steps_per_day
    time_indices = [start_date + timedelta(minutes=10 * t) for t in range(T)]

    # 4. Generate Baseline Velocity Profile V_base(t, N)
    print("Simulating baseline speed profiles...")
    V_free = 50.0
    V_base = np.zeros((T, N))
    
    np.random.seed(42)
    for t in range(T):
        t_day = t % steps_per_day
        dow = (t // steps_per_day) % 7
        is_weekend = dow in [5, 6]
        
        # Rush hour sinusoidal markers
        # Peak 1: 08:30 - 10:30 (Centered around step 57)
        peak1 = np.exp(-((t_day - 57) / 12) ** 2)
        # Peak 2: 17:30 - 20:30 (Centered around step 111)
        peak2 = np.exp(-((t_day - 111) / 15) ** 2)
        
        weekend_factor = 0.5 if is_weekend else 1.0
        
        for n in range(N):
            # Dynamic drift per corridor
            phase_shift = (n * 2) % 6
            t_shifted = t_day + phase_shift
            p1 = np.exp(-((t_shifted - 57) / 12) ** 2)
            p2 = np.exp(-((t_shifted - 111) / 15) ** 2)
            
            v = V_free - (15.0 * p1 + 22.0 * p2) * weekend_factor
            # Add stochastic sin waves and gaussian noise
            v += 2.0 * np.sin(2 * np.pi * t / (steps_per_day * 7)) # Weekly wave
            v += np.random.normal(0, 1.2)
            V_base[t, n] = np.clip(v, 12.0, 60.0)

    # 5. Process Astram Event Data
    print("Reading and mapping Astram events...")
    df = pd.read_csv(csv_path)
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['end_datetime'] = pd.to_datetime(df['end_datetime'], errors='coerce')

    # Event Degradation Loss (Structural Shock Drag)
    L_event = np.zeros((T, N))
    
    # Active events per corridor and day tracker (used for EOD marker counts)
    eod_counts = np.zeros((T, N))

    # Keep track of active events for dynamic graph calculations
    # List of tuples: (start_idx, end_idx, corridor_idx, severity)
    active_events_list = []

    print("Mapping events to timelines and calculating structural shocks...")
    for idx, row in df.iterrows():
        c_name = row['corridor']
        if pd.isna(c_name) or c_name == 'nan':
            # Map geographically based on lat/lon
            lat, lon = row['latitude'], row['longitude']
            if pd.isna(lat) or pd.isna(lon):
                c_name = 'Non-corridor'
            else:
                # Find closest corridor
                best_c = 'Non-corridor'
                best_d = float('inf')
                for c_cand, coord in coords.items():
                    if c_cand == 'Non-corridor':
                        continue
                    d = np.sqrt((lat - coord[0])**2 + (lon - coord[1])**2)
                    if d < best_d:
                        best_d = d
                        best_c = c_cand
                c_name = best_c

        if c_name not in corridor_to_idx:
            c_name = 'Non-corridor'

        c_idx = corridor_to_idx[c_name]
        
        # Timestamps mapping
        start_dt = row['start_datetime']
        if pd.isna(start_dt):
            continue
            
        end_dt = row['end_datetime']
        priority = str(row['priority']).lower()
        
        # Calculate severity S
        if 'high' in priority:
            s_base = 1.0
        elif 'medium' in priority:
            s_base = 0.6
        else:
            s_base = 0.3
            
        if bool(row['requires_road_closure']):
            s_base = min(1.0, s_base + 0.3)
            
        # Default duration if missing
        if pd.isna(end_dt):
            dur_hours = 6 if s_base >= 0.8 else (4 if s_base >= 0.5 else 2)
            end_dt = start_dt + timedelta(hours=dur_hours)
            
        # Map back to timeline grid indices
        t_start_idx = int((start_dt.tz_localize(None) - start_date).total_seconds() // 600)
        t_end_idx = int((end_dt.tz_localize(None) - start_date).total_seconds() // 600)
        
        # Align within timelines
        t_start_idx = max(0, min(T - 1, t_start_idx))
        t_end_idx = max(0, min(T - 1, t_end_idx))
        
        if t_end_idx < t_start_idx:
            t_end_idx, t_start_idx = t_start_idx, t_end_idx
        if t_start_idx == t_end_idx:
            t_end_idx += 1
            
        active_events_list.append((t_start_idx, t_end_idx, c_idx, s_base))
        
        # Compute event-induced traffic speed degradation (Exponential decay overlay)
        # L_event = 25.0 * S * exp(-0.15 * delta_t_steps)
        for t_step in range(t_start_idx, min(T, t_end_idx + 12)): # extend 2 hours past recovery
            delta_t = t_step - t_start_idx
            decay = np.exp(-0.15 * delta_t)
            drag = 25.0 * s_base * decay
            if t_step < T:
                L_event[t_step, c_idx] = max(L_event[t_step, c_idx], drag)
                eod_counts[t_step, c_idx] = min(3.0, eod_counts[t_step, c_idx] + s_base)

    # 6. Final Traffic Speed
    print("Applying event degradation shock to baseline velocities...")
    V = np.maximum(5.0, V_base - L_event)

    # 7. Generate Dynamic Adjacency Tensor A_dynamic (T, N, N)
    print("Generating dynamic, event-conditioned network topologies (Dynamic Graph Orbit)...")
    A_dynamic = np.repeat(A_static[np.newaxis, :, :], T, axis=0)

    # Apply warp factors on events
    # Sort active events by time to process sequentially
    for t_start, t_end, c_idx, severity in active_events_list:
        for t_step in range(t_start, t_end):
            if t_step >= T:
                break
            # Reduce corridor connection capacity
            # A_t,i,j = A_static * (1 - 0.7 * severity)
            # A_t,k,i = A_static * (1 - 0.7 * severity)
            factor_self = 1.0 - 0.7 * severity
            A_dynamic[t_step, c_idx, :] *= factor_self
            A_dynamic[t_step, :, c_idx] *= factor_self
            
            # Neighbor spillover capacity drop (1 - 0.3 * severity)
            factor_neigh = 1.0 - 0.3 * severity
            neighbors = np.where(A_static[c_idx] > 0)[0]
            for n_idx in neighbors:
                A_dynamic[t_step, n_idx, :] *= factor_neigh
                A_dynamic[t_step, :, n_idx] *= factor_neigh

    # 8. Normalize Variables and Time Markers
    print("Normalizing features and exporting files...")
    mean_speed = V.mean(axis=0)
    std_speed = V.std(axis=0)
    std_speed[std_speed == 0] = 1.0
    
    norm_var = (V - mean_speed) / std_speed
    
    # Construct time marker matrix
    norm_time_marker = np.zeros((T, 5))
    for t in range(T):
        t_day = t % steps_per_day
        dow = (t // steps_per_day) % 7
        dom = ((t // steps_per_day) % 30)
        doy = ((t // steps_per_day) % 365)
        
        # Scaling matching easytsf expectation
        norm_time_marker[t, 0] = t_day / (steps_per_day - 1)
        norm_time_marker[t, 1] = dow / 6.0
        norm_time_marker[t, 2] = dom / 29.0
        norm_time_marker[t, 3] = doy / 364.0
        # EOD feature is the local average event severity/count normalized
        norm_time_marker[t, 4] = np.mean(eod_counts[t]) / 3.0

    # Create target directory
    output_dir = r"dataset/AstramBengaluru"
    os.makedirs(output_dir, exist_ok=True)

    # Save features
    np.savez(
        os.path.join(output_dir, "feature.npz"),
        norm_var=norm_var,
        norm_time_marker=norm_time_marker
    )

    # Save scaler statistics
    np.savez(
        os.path.join(output_dir, "var_scaler_info.npz"),
        mean=mean_speed,
        std=std_speed
    )

    # Save static adjacency matrix
    np.save(os.path.join(output_dir, "adj_mat.npy"), A_static)
    
    # Save dynamic adjacency tensor
    np.save(os.path.join(output_dir, "adj_mat_dynamic.npy"), A_dynamic)

    # Save mapping coordinates for dashboard
    coords_df = pd.DataFrame([
        {"corridor": c, "latitude": coords[c][0], "longitude": coords[c][1]}
        for c in CORRIDORS
    ])
    coords_df.to_csv(os.path.join(output_dir, "corridor_coordinates.csv"), index=False)

    print("Pipeline features generated successfully!")
    print(f"norm_var shape: {norm_var.shape}")
    print(f"norm_time_marker shape: {norm_time_marker.shape}")
    print(f"adj_mat shape: {A_static.shape}")
    print(f"adj_mat_dynamic shape: {A_dynamic.shape}")
    print("--- prepare_astram_dataset.py execution completed ---")

if __name__ == "__main__":
    main()
