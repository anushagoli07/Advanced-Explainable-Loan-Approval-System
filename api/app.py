from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.validate import validate_loan_input
from src.features import engineer_features
from src.explain import LoanExplainer
from src.logger import log_prediction

app = FastAPI(title="Explainable Loan Approval API")

# Initialize explainer (it loads the model)
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')
explainer = LoanExplainer(MODEL_DIR)

class LoanApplication(BaseModel):
    income: float
    credit_score: int
    debt_amount: float
    employment_years: int
    age: int

@app.get("/health")
def health():
    return {"status": "ok", "project": "Explainable Loan AI"}

@app.post("/predict")
def predict_loan(app_data: LoanApplication):
    try:
        data_dict = app_data.dict()
        
        # 1. Validate
        validate_loan_input(data_dict)
        
        # 2. Feature Engineering
        input_df = pd.DataFrame([data_dict])
        X_engineered = engineer_features(input_df)
        
        # 3. Scale
        X_scaled = explainer.scaler.transform(X_engineered)
        
        # 4. Predict
        prob = explainer.model.predict_proba(X_scaled)[:, 1][0]
        decision = "APPROVED" if prob > 0.5 else "REJECTED"
        
        # 5. Explain
        shap_vals = explainer.get_explanation(X_scaled)[0]
        feature_names = explainer.get_feature_names()
        
        explanations = []
        for name, val in zip(feature_names, shap_vals):
            # Ensuring val is a scalar for the comparison to avoid ambiguous truth value errors
            scalar_val = float(val.item() if hasattr(val, 'item') else val)
            impact = "positive" if scalar_val > 0 else "negative"
            explanations.append({
                "feature": name,
                "impact_score": round(scalar_val, 4),
                "direction": impact
            })
            
        # 6. Log
        log_prediction(data_dict, decision, prob)
        
        return {
            "decision": decision,
            "approval_probability": round(float(prob), 4),
            "explanations": sorted(explanations, key=lambda x: abs(x['impact_score']), reverse=True)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")
