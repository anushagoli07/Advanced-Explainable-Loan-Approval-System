import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def simulate_monitoring():
    """
    Simulates a monitoring dashboard by generating drift and performance history.
    Detects if credit scores or income values are shifting over time.
    """
    # 1. Prediction Drifts
    # Mock history of approval rates
    history_days = 30
    approval_history = np.random.normal(0.4, 0.05, history_days).tolist()
    
    # 2. Feature Drift (Simulation)
    # Compare training data distribution vs 'live' data
    drift_status = {
        "credit_score_drift": 0.02, # Small KL divergence
        "income_drift": 0.08,        # Significant jump
        "alert_level": "Low"
    }
    
    report = {
        "last_monitored": datetime.now().isoformat(),
        "daily_approval_rates": approval_history,
        "feature_drifts": drift_status,
        "active_alerts": ["Income values are trending higher than training range"]
    }
    
    # Save report for dashboard
    project_root = os.path.dirname(os.path.dirname(__file__))
    monitor_path = os.path.join(project_root, 'data', 'monitor_report.json')
    with open(monitor_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    return report

if __name__ == "__main__":
    print("Monitoring simulation complete.")
    simulate_monitoring()
