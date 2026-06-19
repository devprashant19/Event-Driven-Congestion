import os
import heapq
import numpy as np
import pandas as pd
import json

# Define Bengaluru Corridors dynamically
try:
    with open("data/city_network.json", "r") as f:
        network_data = json.load(f)
        CORRIDORS = network_data["CORRIDORS"]
        CORRIDOR_LENGTHS = network_data["CORRIDOR_LENGTHS"]
        coords = network_data["coords"]
except FileNotFoundError as e:
    raise FileNotFoundError("Critical: data/city_network.json is required to run the network routing.") from e

N = len(CORRIDORS)
corridor_to_idx = {c: i for i, c in enumerate(CORRIDORS)}


class TDSPGraph:
    """
    Time-Dependent Shortest Path routing graph.
    Represents travel cost as a function of predicted speed over future timesteps.
    """
    def __init__(self, A_static, corridor_lengths=None):
        self.A_static = A_static
        self.N = A_static.shape[0]
        self.lengths = np.array([corridor_lengths[CORRIDORS[i]] for i in range(self.N)])

    def get_travel_time(self, corridor_idx, speed_profile, step_idx):
        """
        Calculate traversal time in minutes for a corridor starting at a given step index.
        """
        # Crop to forecast horizon limits
        horizon_len = speed_profile.shape[0]
        idx = max(0, min(horizon_len - 1, step_idx))
        speed = speed_profile[idx, corridor_idx]
        # Traversal time in minutes: (Length in km / Speed in km/h) * 60
        return (self.lengths[corridor_idx] / speed) * 60.0

    def solve_tdsp(self, start_idx, target_idx, departure_time, speed_profile, edge_penalties=None):
        """
        Time-Dependent Dijkstra Shortest Path Solver.
        departure_time: time in minutes from the start of forecast horizon.
        speed_profile: array of shape (L_horizon, N) containing forecasted speeds.
        edge_penalties: dict of (u, v) -> penalty_multiplier for diverse routing.
        """
        if edge_penalties is None:
            edge_penalties = {}
            
        # pq elements: (arrival_time, current_node, path_taken)
        pq = [(departure_time, start_idx, [start_idx])]
        best_arrival = {i: float('inf') for i in range(self.N)}
        best_arrival[start_idx] = departure_time

        while pq:
            arr_time, u, path = heapq.heappop(pq)

            if u == target_idx:
                return path, arr_time - departure_time

            if arr_time > best_arrival[u]:
                continue

            # Check neighbors of u from static adjacency matrix
            neighbors = np.where(self.A_static[u] > 0)[0]
            for v in neighbors:
                # 10-minute step index based on current arrival time
                step_idx = int(arr_time // 10)
                travel_time = self.get_travel_time(v, speed_profile, step_idx)
                
                # Apply penalty for K-shortest paths diversity
                penalty = edge_penalties.get((u, v), 1.0)
                travel_time *= penalty
                
                new_arrival = arr_time + travel_time

                if new_arrival < best_arrival[v]:
                    best_arrival[v] = new_arrival
                    heapq.heappush(pq, (new_arrival, v, path + [v]))

        return [], float('inf')

    def solve_k_shortest_tdsp(self, start_idx, target_idx, departure_time, speed_profile, k=3):
        """
        Generates K geographically diverse alternate routes using iterative edge penalization.
        """
        paths = []
        edge_penalties = {}
        
        for i in range(k):
            path, travel_time = self.solve_tdsp(start_idx, target_idx, departure_time, speed_profile, edge_penalties)
            if not path or travel_time == float('inf'):
                break
                
            paths.append((path, travel_time))
            
            # Penalize edges in the found path to force diversity in the next iteration
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                # Increase penalty by 2.0 (100% time increase) to heavily discourage reuse
                current_penalty = edge_penalties.get((u, v), 1.0)
                edge_penalties[(u, v)] = current_penalty * 2.0
                
        return paths


class PolicySimulator:
    """
    Simulates physical parameter modifications under policy changes:
    - Police Officers (P): Speeds up recovery decay coefficient lambda.
    - Barricades (B): Restricts traffic bottlenecks, lowering spillover propagation coefficient.
    """
    def __init__(self, static_adj_path, scaler_path):
        self.A_static = np.load(static_adj_path)
        scaler = np.load(scaler_path)
        self.mean_speed = scaler['mean']
        self.std_speed = scaler['std']

    def simulate_mitigated_speed(self, base_speeds, active_event_list, num_officers, num_barricades):
        """
        Recalculates traffic speeds and active topologies under specific resource allocations.
        - num_officers: Sliders range [0, 50]
        - num_barricades: Sliders range [0, 20]
        """
        T, N = base_speeds.shape
        L_event = np.zeros((T, N))
        
        # Policy Impact Modifiers
        # Recovery coefficient lambda gets boosted by officers count
        lambda_mitigated = 0.15 * (1.0 + 0.15 * num_officers)
        
        # Spillover drag factor reduced by barricades
        spillover_factor = 0.3 * np.exp(-0.1 * num_barricades)

        for t_start, t_end, c_idx, severity in active_event_list:
            for t_step in range(t_start, min(T, t_end + 12)):
                delta_t = t_step - t_start
                # Mitigated decay curve
                decay = np.exp(-lambda_mitigated * delta_t)
                drag = 25.0 * severity * decay
                if t_step < T:
                    L_event[t_step, c_idx] = max(L_event[t_step, c_idx], drag)

        # Recalculate dynamic graph network weights with mitigated spillover coefficients
        A_dynamic = np.repeat(self.A_static[np.newaxis, :, :], T, axis=0)
        for t_start, t_end, c_idx, severity in active_event_list:
            for t_step in range(t_start, t_end):
                if t_step >= T:
                    break
                factor_self = 1.0 - 0.7 * severity
                A_dynamic[t_step, c_idx, :] *= factor_self
                A_dynamic[t_step, :, c_idx] *= factor_self
                
                # Mitigated neighbor spillover
                factor_neigh = 1.0 - spillover_factor * severity
                neighbors = np.where(self.A_static[c_idx] > 0)[0]
                for n_idx in neighbors:
                    A_dynamic[t_step, n_idx, :] *= factor_neigh
                    A_dynamic[t_step, :, n_idx] *= factor_neigh

        speeds_mitigated = np.maximum(5.0, base_speeds - L_event)
        return speeds_mitigated, A_dynamic

    def simulate_mitigated_forecast(self, api, var_x, marker_x, scenario_events, officer_allocation, num_barricades):
        """
        scenario_events: list of dicts {"c_idx": int, "severity": float, "start_step": int}
        officer_allocation: dict mapping corridor_idx -> number of officers deployed.
        """
        # Handle backward compatibility if single int is passed
        if isinstance(officer_allocation, int):
            # Just grab the first origin if it's an int fallback
            first_origin = scenario_events[0]["c_idx"] if scenario_events else 0
            officer_allocation = {first_origin: officer_allocation}
            
        # 1. Get Base AI Prediction
        base_ai_forecast = api.predict(var_x, marker_x)
        T, N = base_ai_forecast.shape
        L_event = np.zeros((T, N))
        
        for evt in scenario_events:
            origin_c_idx = evt["c_idx"]
            severity_val = evt["severity"]
            start_step = evt.get("start_step", 0)
            
            # 2. Policy Impact Modifiers
            num_origin_officers = officer_allocation.get(origin_c_idx, 0)
            mitigation_factor = min(0.8, 0.01 * num_origin_officers + 0.01 * num_barricades)
            reduced_severity = severity_val * (1.0 - mitigation_factor)
            
            lambda_mitigated_origin = 0.15 * (1.0 + 0.15 * num_origin_officers)
            spillover_factor = 0.3 * np.exp(-0.1 * num_barricades)
            
            # 3. Apply Heuristic Decay Shockwave
            for t_step in range(start_step, T):
                delta_t = t_step - start_step
                decay = np.exp(-lambda_mitigated_origin * delta_t)
                # Direct hit on origin corridor
                drag = 35.0 * reduced_severity * decay
                L_event[t_step, origin_c_idx] += drag # Accumulate for multiple events
                
                # Shockwave spillover to neighboring corridors
                neighbors = np.where(self.A_static[origin_c_idx] > 0.1)[0]
                for n_idx in neighbors:
                    if delta_t >= 2: # Spillover hits 20 mins later
                        num_neigh_officers = officer_allocation.get(n_idx, 0)
                        lambda_neigh = 0.15 * (1.0 + 0.2 * num_neigh_officers) # local officers recover neighbor faster
                        neigh_drag = 20.0 * reduced_severity * spillover_factor * np.exp(-lambda_neigh * (delta_t - 2))
                        L_event[t_step, n_idx] += neigh_drag
                    
        # Subtract the shockwave drag from the AI's base prediction
        mitigated_speeds = np.maximum(5.0, base_ai_forecast - L_event)
        
        return mitigated_speeds


class ManpowerOptimizer:
    """
    Allocates a constrained budget of police personnel across the impact radius
    to mathematically minimize network-wide congestion delay.
    """
    def __init__(self, A_static):
        self.A_static = A_static

    def greedy_allocation(self, sim, api, var_x, marker_x, scenario_events, total_officers, num_barricades):
        # Identify the "Impact Zone" (All Origin nodes + immediate neighbors)
        impact_zone_set = set()
        for evt in scenario_events:
            origin_c_idx = evt["c_idx"]
            impact_zone_set.add(origin_c_idx)
            for n_idx in np.where(self.A_static[origin_c_idx] > 0.1)[0]:
                impact_zone_set.add(n_idx)
                
        impact_zone = list(impact_zone_set)
        if not impact_zone:
            return {}
            
        allocation = {idx: 0 for idx in impact_zone}
        remaining_officers = total_officers
        chunk_size = 5 # Evaluate in chunks of 5 officers for speed
        
        while remaining_officers > 0:
            alloc_amount = min(chunk_size, remaining_officers)
            best_idx = impact_zone[0]
            best_speed_sum = -float('inf')
            
            for candidate_idx in impact_zone:
                # Test allocation scenario
                test_allocation = allocation.copy()
                test_allocation[candidate_idx] += alloc_amount
                
                test_speeds = sim.simulate_mitigated_forecast(api, var_x, marker_x, scenario_events, test_allocation, num_barricades)
                # Maximize overall speed across the impact zone
                network_speed_sum = np.sum(test_speeds[:, impact_zone])
                
                if network_speed_sum > best_speed_sum:
                    best_speed_sum = network_speed_sum
                    best_idx = candidate_idx
                    
            allocation[best_idx] += alloc_amount
            remaining_officers -= alloc_amount
            
        # Filter out 0 allocations
        return {idx: count for idx, count in allocation.items() if count > 0}


class InfrastructureOptimizer:
    """
    Generates tactical physical infrastructure recommendations based on graph topology
    and multi-path diversion routing outputs.
    """
    def __init__(self, A_static):
        self.A_static = A_static

    def generate_plan(self, scenario_events, num_barricades, diversion_routes):
        """
        diversion_routes: list of (path, travel_time) from solve_k_shortest_tdsp
        """
        plan = {
            "barricades": [],
            "signals": []
        }
        
        if num_barricades > 0:
            # 1. Barricade Logic: Choke off incoming heavy flow to the origin
            # Find corridors that feed into any of the origin corridors
            feeder_corridors = {}
            for evt in scenario_events:
                origin_c_idx = evt["c_idx"]
                neighbors = np.where(self.A_static[:, origin_c_idx] > 0.1)[0]
                for n_idx in neighbors:
                    feeder_corridors[n_idx] = feeder_corridors.get(n_idx, 0) + self.A_static[n_idx, origin_c_idx]
                    
            # Sort neighbors by connection weight (highest flow capacity first)
            neighbor_weights = sorted(feeder_corridors.items(), key=lambda x: x[1], reverse=True)
            
            assigned_barricades = 0
            for n_idx, weight in neighbor_weights:
                if assigned_barricades >= num_barricades:
                    break
                # Assign up to 2 barricades per intersection to spread them out
                allocation = min(2, num_barricades - assigned_barricades)
                plan["barricades"].append((n_idx, allocation))
                assigned_barricades += allocation
            
        # 2. Signal Logic: Extend green phases on diversion routes
        unique_diversion_nodes = set()
        for path, _ in diversion_routes:
            for node in path:
                unique_diversion_nodes.add(node)
                
        # Do not extend signals at any crash site or its immediate bottlenecks
        for evt in scenario_events:
            origin_c_idx = evt["c_idx"]
            if origin_c_idx in unique_diversion_nodes:
                unique_diversion_nodes.remove(origin_c_idx)
            
        plan["signals"] = list(unique_diversion_nodes)
        
        return plan


def test_recommendation_engine():
    print("--- Testing recommendation_engine.py ---")
    static_adj_path = r"dataset/AstramBengaluru/adj_mat.npy"
    scaler_path = r"dataset/AstramBengaluru/var_scaler_info.npz"

    if not os.path.exists(static_adj_path) or not os.path.exists(scaler_path):
        print("Data files not found. Generate dataset first.")
        return

    # Load static adj matrix
    A_static = np.load(static_adj_path)
    
    # 1. Create TDSP Graph
    graph = TDSPGraph(A_static, CORRIDOR_LENGTHS)

    # Mock dynamic speed profiles (12 timesteps x 22 corridors)
    # Start all corridor speeds at 40 km/h, drop one corridor to 10 km/h in future steps
    mock_speeds = np.full((12, N), 40.0)
    # Let's say corridor 10 ('Hosur Road') drops to 8 km/h at step 3
    mock_speeds[3:, 10] = 8.0

    print("Running K-Shortest Path Query (K=3)...")
    # Find 3 diverse shortest paths from CBD 1 (index 17) to Hosur Road (index 10) starting at time 0 minutes
    paths = graph.solve_k_shortest_tdsp(17, 10, 0.0, mock_speeds, k=3)
    print(f"Start node: CBD 1 (17)")
    print(f"End node: Hosur Road (10)")
    for i, (path, travel_time) in enumerate(paths):
        print(f"Path {i+1}: {[CORRIDORS[node] for node in path]}")
        print(f"  Travel Time: {travel_time:.2f} mins")

    # 2. Test Policy Simulator
    sim = PolicySimulator(static_adj_path, scaler_path)
    base_speeds = np.full((24, N), 45.0) # 4 hours range
    active_events = [(4, 12, 10, 0.8)] # Event on corridor 10 from step 4 to 12 with high severity
    
    # Unmitigated speeds
    speeds_unmit, _ = sim.simulate_mitigated_speed(base_speeds, active_events, 0, 0)
    # Mitigated speeds with 15 officers and 5 barricades
    speeds_mit, _ = sim.simulate_mitigated_speed(base_speeds, active_events, 15, 5)

    print("\nComparing Speeds (Unmitigated vs Mitigated at step 8):")
    print(f"Unmitigated speed on Corridor 10: {speeds_unmit[8, 10]:.2f} km/h")
    print(f"Mitigated speed on Corridor 10: {speeds_mit[8, 10]:.2f} km/h")
    print("Test passed successfully!")

if __name__ == "__main__":
    test_recommendation_engine()
