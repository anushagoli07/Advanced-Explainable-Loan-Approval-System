import pandas as pd

def engineer_features(df):
    """
    Transforms raw applicant data into meaningful ML features.
    Calculates Debt-to-Income (DTI) and Employment Stability Index.
    """
    df = df.copy()
    
    # Feature 1: Debt-to-Income Ratio (DTI)
    # Avoid division by zero
    df['dti_ratio'] = df['debt_amount'] / (df['income'] + 1)
    
    # Feature 2: Employment Stability Index
    # Combination of age and employment years
    df['stability_index'] = df['employment_years'] / (df['age'] - 17)
    
    # Select final features for model training
    features = ['income', 'credit_score', 'debt_amount', 'employment_years', 'age', 'dti_ratio', 'stability_index']
    return df[features]
