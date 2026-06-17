import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation_engine import CORRIDORS, CORRIDOR_LENGTHS

class TrafficMetrics:
    """
    Translates raw speed predictions (km/h) into actionable traffic engineering metrics
    like Congestion Severity and Delay Times.
    """
    def __init__(self, v_free=50.0):
        self.v_free = v_free
        # Corridor lengths in km, shape: (N,)
        self.lengths = np.array([CORRIDOR_LENGTHS[c] for c in CORRIDORS])

    def calculate_metrics(self, v_forecast):
        """
        Calculates congestion severity and delay times from forecasted speeds.
        
        Args:
            v_forecast: numpy array of shape (pred_len, N) or (N,), predicted speeds in km/h.
            
        Returns:
            severity: numpy array of the same shape, congestion severity in % (0-100%).
            delay: numpy array of the same shape, expected delay time in minutes.
        """
        # Ensure speeds are positive to avoid division by zero (min 1 km/h)
        v_f = np.maximum(v_forecast, 1.0)
        
        # 1. Congestion Severity (%)
        # Drop from free-flow speed, capped between 0% and 100%
        severity = np.clip((self.v_free - v_f) / self.v_free * 100.0, 0.0, 100.0)
        
        # 2. Delay Time (minutes)
        # Travel time (mins) = (length_km / speed_kmh) * 60
        tt_forecast = (self.lengths / v_f) * 60.0
        tt_free = (self.lengths / self.v_free) * 60.0
        
        # Delay is the extra time spent compared to free-flow conditions
        delay = np.maximum(tt_forecast - tt_free, 0.0)
        
        return severity, delay

if __name__ == "__main__":
    # Quick unit test
    metrics = TrafficMetrics(v_free=50.0)
    
    # Mock speed: [Free flow, Mild traffic, Heavy traffic] on the first 3 corridors
    mock_speeds = np.array([50.0, 30.0, 10.0])
    # The lengths of the first 3 corridors are: 12.0, 15.0, 8.0 km respectively
    
    # Subset to test just the first 3 nodes
    metrics.lengths = metrics.lengths[:3]
    
    severity, delay = metrics.calculate_metrics(mock_speeds)
    print("--- Traffic Metrics Test ---")
    print(f"Speeds (km/h): {mock_speeds}")
    print(f"Severity (%):  {severity}")
    print(f"Delay (mins):  {delay}")
