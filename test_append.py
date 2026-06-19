import json
import time
import numpy as np

# Create dummy speeds_mit (12, 22) array
dummy_speeds = np.ones((12, 22)) * 40.0

new_event = {
    "id": 9999,
    "name": "Test Event",
    "speeds_mit": dummy_speeds.tolist(),
    "events": [],
    "officers_deployed": 10,
    "barricades": 2,
    "timestamp": time.strftime('%Y-%m-%d %H:%M')
}

with open("database/historical_events.json", "w") as f:
    json.dump([new_event], f)

print("Mock event saved with ID 9999.")
