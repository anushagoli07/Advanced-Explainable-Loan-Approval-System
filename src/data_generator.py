import pandas as pd
import numpy as np
import os

def generate_loan_data(filepath, n_samples=2000, random_seed=42):
    """
    Generates a synthetic loan application dataset.
    Features: income, credit_score, debt_amount, employment_years, age, gender (for bias check).
    """
    np.random.seed(random_seed)
    
    # Base features
    income = np.random.normal(60000, 20000, n_samples).clip(20000, 200000)
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    debt_amount = np.random.normal(15000, 10000, n_samples).clip(0, 100000)
    employment_years = np.random.randint(0, 40, n_samples)
    age = np.random.randint(18, 75, n_samples)
    
    # Sensitive attribute for bias check (Binary Gender simulation)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    
    # Target: Loan Approved (1) or Rejected (0)
    # Logic: High credit score + High income + Low debt + High employment = Approval
    # We add some noise and a slight weight to 'income' for later fairness checking
    scores = (
        0.4 * (credit_score / 850) +
        0.3 * (income / 100000) -
        0.2 * (debt_amount / 50000) +
        0.1 * (employment_years / 20) +
        np.random.normal(0, 0.05, n_samples)
    )
    
    approved = (scores > 0.5).astype(int)
    
    df = pd.DataFrame({
        'income': income.round(2),
        'credit_score': credit_score.astype(int),
        'debt_amount': debt_amount.round(2),
        'employment_years': employment_years,
        'age': age,
        'gender': gender,
        'approved': approved
    })
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Synthetic loan dataset generated: {filepath}")
    return df

if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'loan_data.csv')
    generate_loan_data(filepath)
