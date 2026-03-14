def validate_loan_input(data):
    """
    Validates loan application inputs to prevent erroneous predictions.
    Checks for positive income, valid credit score range, and realistic employment history.
    """
    if data.get("income", 0) <= 0:
        raise ValueError("Annual Income must be a positive number.")
    
    credit_score = data.get("credit_score", 0)
    if not (300 <= credit_score <= 850):
        raise ValueError("Credit Score must be between 300 and 850.")
    
    if data.get("employment_years", -1) < 0:
        raise ValueError("Employment Years cannot be negative.")
        
    if data.get("debt_amount", -1) < 0:
        raise ValueError("Debt Amount cannot be negative.")
        
    return True
