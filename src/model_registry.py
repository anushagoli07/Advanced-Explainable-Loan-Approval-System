import joblib
import os
import json
from datetime import datetime

class ModelRegistry:
    def __init__(self, base_path):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
    def save_model(self, model, scaler, metrics, version="v1"):
        """
        Saves model artifacts and performance metadata to a versioned directory.
        """
        version_dir = os.path.join(self.base_path, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save Binaries
        joblib.dump(model, os.path.join(version_dir, f'model_{version}.pkl'))
        joblib.dump(scaler, os.path.join(version_dir, f'scaler_{version}.pkl'))
        
        # Save Metrics & Metadata
        metadata = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        with open(os.path.join(version_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model version {version} registered in {version_dir}")
        
    def load_latest(self, version="v1"):
        """
        Loads the specified model version.
        """
        version_dir = os.path.join(self.base_path, version)
        model = joblib.load(os.path.join(version_dir, f'model_{version}.pkl'))
        scaler = joblib.load(os.path.join(version_dir, f'scaler_{version}.pkl'))
        return model, scaler

if __name__ == "__main__":
    # Test
    registry = ModelRegistry("c:\\Users\\phani\\OneDrive\\Documents\\gat\\explainable-loan-approval\\models")
    print("Registry initialized.")
