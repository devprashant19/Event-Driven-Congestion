import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from easytsf.runner.traffic_forecasting_runner import Runner

class LiveInferenceAPI:
    """
    Live Inference Wrapper for PyTorch Lightning Forecasting Models.
    Connects to MockTrafficStream or live streaming interfaces to provide real-time 
    traffic speed forecasts.
    """
    def __init__(self, checkpoint_path, device=None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            
        print(f"Loading Live Inference API from {checkpoint_path}...")
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Load the PyTorch Lightning Runner from the checkpoint
        self.runner = Runner.load_from_checkpoint(checkpoint_path)
        
        # Set to evaluation mode and move to device
        self.runner.eval()
        self.runner.to(self.device)
        self.runner.freeze()  # Freeze gradients for fast inference
        
        print(f"Model loaded successfully on {self.device}!")

    def predict(self, var_x, marker_x):
        """
        Takes raw historical traffic window and predicts the future horizon.
        
        Args:
            var_x: numpy array of shape (L, N, 1), normalized historical speeds.
            marker_x: numpy array of shape (L, N, 5), time and event markers.
            
        Returns:
            rescaled_prediction: numpy array of shape (pred_len, N), forecasted speeds in km/h.
        """
        # Convert to torch tensors, add batch dimension, and move to device
        tensor_var_x = torch.tensor(var_x, dtype=torch.float32).unsqueeze(0).to(self.device)
        tensor_marker_x = torch.tensor(marker_x, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Model forward pass
            # Note: the runner expects (var_x, marker_x)
            prediction = self.runner.model(tensor_var_x, tensor_marker_x)
            
            # Slice to only get the predicted length if it outputs the full sequence
            prediction = prediction[:, -self.runner.hparams.pred_len:, :]
            
            # Inverse transform the normalized prediction back to km/h speeds
            rescaled_prediction = self.runner.inverse_transform_var(prediction)
            
        # Return as numpy array, removing the batch dimension
        return rescaled_prediction.squeeze(0).cpu().numpy()

if __name__ == "__main__":
    # Test block to verify LiveInferenceAPI works with MockTrafficStream
    import time
    from script.mock_stream import MockTrafficStream
    from script.traffic_metrics import TrafficMetrics
    
    # Example checkpoint (update if a different model is trained)
    ckpt_path = "save/STIDEF_AstramBengaluru/22072b82a2/seed_0/checkpoints/epoch=0-step=492.ckpt"
    
    try:
        api = LiveInferenceAPI(ckpt_path)
        stream = MockTrafficStream()
        metrics_engine = TrafficMetrics(v_free=50.0)
        
        print("\n--- Running Live Inference Test ---")
        for i in range(2):
            var_x, marker_x, step = stream.stream_next()
            if step == -1:
                break
                
            start_time = time.time()
            prediction = api.predict(var_x, marker_x)
            latency = (time.time() - start_time) * 1000
            
            print(f"Frame {i+1} (Step {step}):")
            print(f"  Input Shape:  var_x {var_x.shape}, marker_x {marker_x.shape}")
            print(f"  Output Shape: {prediction.shape} (pred_len, nodes)")
            print(f"  Inference Latency: {latency:.2f} ms")
            
            # Calculate and print metrics
            severity, delay = metrics_engine.calculate_metrics(prediction)
            
            avg_speed = prediction.mean()
            max_severity = severity.max()
            max_delay = delay.max()
            
            print(f"  Average Forecasted Speed: {avg_speed:.2f} km/h")
            print(f"  Max Congestion Severity:  {max_severity:.1f}%")
            print(f"  Max Expected Delay:       {max_delay:.1f} mins\n")
            
    except FileNotFoundError as e:
        print(f"\nSkipping test: {e}\nPlease train the model first or provide a valid checkpoint.")
