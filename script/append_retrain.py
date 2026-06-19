import os
import sys
import json
import numpy as np
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_id", type=str, required=True, help="ID of the event to retrain on")
    args = parser.parse_args()

    event_id = int(args.event_id)
    db_path = "database/historical_events.json"
    
    if not os.path.exists(db_path):
        print("Historical events DB not found.")
        sys.exit(1)
        
    with open(db_path, "r") as f:
        events = json.load(f)
        
    target_event = None
    for e in events:
        if e.get("id") == event_id:
            target_event = e
            break
            
    if not target_event:
        print(f"Event ID {event_id} not found in DB.")
        sys.exit(1)
        
    speeds_mit = target_event.get("speeds_mit")
    if not speeds_mit:
        print("Error: No speeds_mit data found in the event.")
        sys.exit(1)
        
    speeds_mit = np.array(speeds_mit) # Shape should be (12, 22)
    
    # 1. Load Scaler
    scaler_path = "dataset/AstramBengaluru/var_scaler_info.npz"
    scaler = np.load(scaler_path)
    mean = scaler["mean"]
    std = scaler["std"]
    
    # 2. Normalize
    norm_new_var = (speeds_mit - mean) / std
    
    # 3. Load existing feature.npz
    feature_path = "dataset/AstramBengaluru/feature.npz"
    features = np.load(feature_path)
    old_norm_var = features["norm_var"]
    old_norm_time_marker = features["norm_time_marker"]
    
    # Append
    new_norm_var = np.vstack([old_norm_var, norm_new_var])
    
    # Just duplicate the last time marker for the new 12 steps for simplicity
    last_marker = old_norm_time_marker[-1:]
    new_markers = np.repeat(last_marker, 12, axis=0)
    
    # Increment the time step for the new markers (10 minutes each step)
    # The first element is Time of Day [0, 1]. There are 144 steps per day.
    for i in range(12):
        new_markers[i, 0] = (new_markers[i, 0] * 143 + 1) / 143.0
        if new_markers[i, 0] > 1.0:
            new_markers[i, 0] = 0.0
    
    new_norm_time_marker = np.vstack([old_norm_time_marker, new_markers])
    
    np.savez(feature_path, norm_var=new_norm_var, norm_time_marker=new_norm_time_marker)
    print(f"Appended 12 timesteps. New norm_var shape: {new_norm_var.shape}")
    
    # 4. Append to adj_mat_dynamic.npy
    adj_dynamic_path = "dataset/AstramBengaluru/adj_mat_dynamic.npy"
    if os.path.exists(adj_dynamic_path):
        old_adj = np.load(adj_dynamic_path)
        # Duplicate the last adj matrix
        last_adj = old_adj[-1:]
        new_adjs = np.repeat(last_adj, 12, axis=0)
        new_adj_dynamic = np.concatenate([old_adj, new_adjs], axis=0)
        np.save(adj_dynamic_path, new_adj_dynamic)
        print(f"Appended to adj_mat_dynamic. New shape: {new_adj_dynamic.shape}")
        
    # 5. Launch Training
    print("Launching background retraining job...")
    # Using STIDEF as the default configuration for the problem statement
    log_file = open("database/retrain.log", "w")
    subprocess.Popen([sys.executable, "train.py", "-c", "config/ASTRAM/STIDEF.py"], stdout=log_file, stderr=subprocess.STDOUT)
    print("Retraining triggered successfully.")

if __name__ == "__main__":
    main()
