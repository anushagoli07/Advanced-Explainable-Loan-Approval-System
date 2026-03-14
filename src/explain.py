import shap
import pandas as pd
import numpy as np
import os
from .model_registry import ModelRegistry

class LoanExplainer:
    def __init__(self, model_dir, version="v1"):
        registry = ModelRegistry(model_dir)
        self.model, self.scaler = registry.load_latest(version)
        # Use a small sample to initialize the explainer (KernelExplainer or TreeExplainer)
        # TreeExplainer is much faster for Random Forest
        self.explainer = shap.TreeExplainer(self.model)
        
    def get_explanation(self, input_features_scaled):
        """
        Calculates SHAP values for a single prediction.
        """
        shap_values = self.explainer.shap_values(input_features_scaled)
        
        # In multi-class, shap_values is a list. For binary with RF, index 1 is 'Approved'
        if isinstance(shap_values, list):
            instance_shap = shap_values[1]
        else:
            instance_shap = shap_values
            
        return instance_shap

    def get_feature_names(self):
        return ['income', 'credit_score', 'debt_amount', 'employment_years', 'age', 'dti_ratio', 'stability_index']
