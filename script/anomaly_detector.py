import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation_engine import CORRIDORS

class AnomalyDetector:
    """
    Detects sudden unplanned traffic events and predicts the impact radius of events
    using forecasted severity metrics.
    """
    def __init__(self, anomaly_threshold_pct=30.0, impact_threshold_pct=20.0):
        self.anomaly_threshold_pct = anomaly_threshold_pct
        self.impact_threshold_pct = impact_threshold_pct

    def detect_unplanned_events(self, live_severity, active_planned_events=None):
        """
        Detects corridors experiencing severe congestion that is not explained by planned events.
        
        Args:
            live_severity: numpy array (N,) of current congestion severity percentages.
            active_planned_events: list of corridor indices that have known planned events.
            
        Returns:
            anomalies: list of tuples (corridor_index, severity) that are anomalous.
        """
        if active_planned_events is None:
            active_planned_events = []
            
        anomalies = []
        for i, severity in enumerate(live_severity):
            if severity >= self.anomaly_threshold_pct and i not in active_planned_events:
                anomalies.append((i, severity))
                
        return anomalies

    def calculate_impact_radius(self, forecasted_severity, origin_corridors):
        """
        Identifies all corridors that will be impacted by events based on future forecast.
        
        Args:
            forecasted_severity: numpy array (pred_len, N) of forecasted severities.
            origin_corridors: list of corridor indices where the event originated.
            
        Returns:
            impact_zone: list of tuples (corridor_index, max_forecasted_severity) 
                         for corridors outside the origin that breach the impact threshold.
        """
        # Get maximum severity over the forecast horizon for each corridor
        max_severity = forecasted_severity.max(axis=0)  # Shape (N,)
        
        impact_zone = []
        for i, severity in enumerate(max_severity):
            if i not in origin_corridors and severity >= self.impact_threshold_pct:
                impact_zone.append((i, severity))
                
        # Sort by most severely impacted
        impact_zone.sort(key=lambda x: x[1], reverse=True)
        return impact_zone

if __name__ == "__main__":
    # Unit test
    detector = AnomalyDetector()
    
    # Mock current severity: Corridor 2 is 45% (Anomaly), Corridor 5 is 80% (Planned Event)
    live_sev = np.zeros(len(CORRIDORS))
    live_sev[2] = 45.0
    live_sev[5] = 80.0
    
    print("--- Unplanned Event Detection ---")
    anomalies = detector.detect_unplanned_events(live_sev, active_planned_events=[5])
    for idx, sev in anomalies:
        print(f"Detected Anomaly on {CORRIDORS[idx]} (Severity: {sev:.1f}%)")
        
    # Mock forecast severity: Corridor 5 stays bad, spills over to Corridor 4 and 6
    forecast_sev = np.zeros((12, len(CORRIDORS)))
    forecast_sev[:, 5] = 80.0
    forecast_sev[6:, 4] = 35.0  # Spills to node 4 at step 6
    forecast_sev[8:, 6] = 25.0  # Spills to node 6 at step 8
    
    print("\n--- Impact Radius Prediction ---")
    impact_zone = detector.calculate_impact_radius(forecast_sev, origin_corridors=[5])
    for idx, sev in impact_zone:
        print(f"Impacted: {CORRIDORS[idx]} (Max Severity: {sev:.1f}%)")
