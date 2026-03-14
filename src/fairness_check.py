import pandas as pd
import numpy as np
import os

def check_group_fairness(data_path):
    """
    Analyzes model 'fairness' by comparing approval rates across different groups.
    Specifically checks for disparities between Female and Male applicants.
    """
    if not os.path.exists(data_path):
        return "Data not found. Please run training first."
        
    df = pd.read_csv(data_path)
    
    # Calculate approval rates by gender
    gender_stats = df.groupby('gender')['approved'].mean()
    
    # Demographic Parity Ratio: (P(Approved|Female) / P(Approved|Male))
    # Ideally should be close to 1.0 (80% rule is common in industry)
    female_rate = gender_stats.get('Female', 0)
    male_rate = gender_stats.get('Male', 1) # Avoid division by zero
    
    parity_ratio = female_rate / male_rate if male_rate > 0 else 1.0
    
    report = {
        "female_approval_rate": round(female_rate, 4),
        "male_approval_rate": round(male_rate, 4),
        "demographic_parity_ratio": round(parity_ratio, 4),
        "status": "PASS" if 0.8 <= parity_ratio <= 1.25 else "WARN: Potential Bias Detected"
    }
    
    return report

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'loan_data.csv')
    print("Fairness Report:", check_group_fairness(path))
