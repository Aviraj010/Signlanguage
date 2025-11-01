import numpy as np
import os
import json

class Preprocessor:
    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_landmarks(self, label, landmarks):
        """Save landmarks (x, y coordinates) as a JSON file for training"""
        file_path = os.path.join(self.save_dir, f"{label}.json")

        # Convert landmarks (list of tuples) to serializable list
        data = np.array(landmarks).tolist()

        # Load existing data if file exists
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Append new sample (each frame = one sample)
        existing_data.append(data)

        # Save back to file
        with open(file_path, "w") as f:
            json.dump(existing_data, f)

        print(f"✅ Saved sample for label '{label}' — Total samples: {len(existing_data)}")
