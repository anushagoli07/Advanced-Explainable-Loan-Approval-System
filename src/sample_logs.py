import os
import json
from datetime import datetime

def generate_sample_logs():
    """
    Creates a set of realistic prediction logs for portfolio demonstration.
    """
    logs = [
        {
            "timestamp": "2026-03-14T10:00:00",
            "input": {"income": 45000, "credit_score": 580, "debt_amount": 25000, "employment_years": 2, "age": 28},
            "prediction": "REJECTED",
            "probability": 0.32,
            "top_reason": "Low credit score (-0.45), High debt ratio (-0.30)"
        },
        {
            "timestamp": "2026-03-14T10:15:00",
            "input": {"income": 120000, "credit_score": 790, "debt_amount": 5000, "employment_years": 12, "age": 45},
            "prediction": "APPROVED",
            "probability": 0.94,
            "top_reason": "Excellent credit score (+0.55), Stable employment (+0.25)"
        },
        {
            "timestamp": "2026-03-14T11:05:00",
            "input": {"income": 55000, "credit_score": 640, "debt_amount": 8000, "employment_years": 4, "age": 32},
            "prediction": "APPROVED",
            "probability": 0.58,
            "top_reason": "Moderate credit (+0.15), Low debt ratio (+0.20)"
        }
    ]
    
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'sample_gallery.json')
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    print(f"Sample logs generated at {log_file}")

if __name__ == "__main__":
    generate_sample_logs()
