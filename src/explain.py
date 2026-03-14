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
        Supports different SHAP output formats (lists, arrays, Explanation objects).
        """
        shap_result = self.explainer.shap_values(input_features_scaled)
        
        # 1. Handle Explanation objects (shap >= 0.40)
        if hasattr(shap_result, "values"):
            shap_values = shap_result.values
        else:
            shap_values = shap_result

        # 2. Handle Binary Classification formats
        # Random Forest often returns a list [neg_array, pos_array] 
        # or a 3D array (2, samples, features)
        if isinstance(shap_values, list):
            # Take the positive class values if two classes are present
            instance_shap = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Shape is (classes, samples, features), take index 1 if present
            instance_shap = shap_values[1] if shap_values.shape[0] > 1 else shap_values[0]
        else:
            # Already 2D (samples, features) or other format
            instance_shap = shap_values
            
        return np.array(instance_shap)

    def get_feature_names(self):
        return ['income', 'credit_score', 'debt_amount', 'employment_years', 'age', 'dti_ratio', 'stability_index']
