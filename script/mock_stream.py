import os
import time
import numpy as np

class MockTrafficStream:
    """
    Simulates a real-time live traffic data stream by yielding data frame-by-frame 
    from the generated dataset. This is the interface the Dashboard and live models 
    will consume.
    """
    def __init__(self, data_dir="dataset/AstramBengaluru", history_len=12):
        self.data_dir = data_dir
        self.history_len = history_len
        
        feature_path = os.path.join(data_dir, "feature.npz")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature dataset not found at {feature_path}. Run prepare_astram_dataset.py first.")
            
        data = np.load(feature_path)
        self.norm_var = data['norm_var']          # (T, N)
        self.norm_time_marker = data['norm_time_marker']  # (T, 5)
        self.T = self.norm_var.shape[0]
        
        # Start streaming from a point where we have enough history
        self.current_step = self.history_len

    def stream_next(self):
        """
        Yields the next available historical window (history_len) and the current timestamp marker.
        Returns:
            var_x: (history_len, N, 1) numpy array of normalized speeds
            marker_x: (history_len, N, 5) numpy array of time markers
            current_step: int
        """
        if self.current_step >= self.T:
            return None, None, -1
            
        # Get historical window [current_step - history_len : current_step]
        var_window = self.norm_var[self.current_step - self.history_len : self.current_step]
        marker_window = self.norm_time_marker[self.current_step - self.history_len : self.current_step]
        
        # Reshape to expected model inputs: 
        # var_x: (L, N, C) where C=1
        # marker_x: (L, N, 5) where 5 is time features
        N = var_window.shape[1]
        var_x = var_window.reshape(self.history_len, N, 1)
        
        # Broadcast marker_window (L, 5) to (L, N, 5)
        marker_x = np.repeat(marker_window[:, np.newaxis, :], N, axis=1)
        
        step_returned = self.current_step
        self.current_step += 1
        
        return var_x, marker_x, step_returned

if __name__ == "__main__":
    print("Initializing Mock Traffic Stream...")
    stream = MockTrafficStream()
    print("Fetching first 3 live frames...")
    
    for i in range(3):
        var_x, marker_x, step = stream.stream_next()
        print(f"\n--- Frame {i+1} (Step {step}) ---")
        print(f"var_x shape: {var_x.shape}")
        print(f"marker_x shape: {marker_x.shape}")
        # Add a small sleep to simulate latency
        time.sleep(0.5)
    
    print("\nStream simulation successful.")
