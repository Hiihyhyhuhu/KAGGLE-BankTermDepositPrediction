import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import cupy as cp
from sklearn.linear_model import LogisticRegression
import cuml

def generate_stacking_shap(model, X_test_meta_df, save_path, prefix):
    """
    Standalone function to handle SHAP explanations for the Stacking Meta-Learner.
    """
    meta_learner = model.meta_learner_model
    
    # 1. Handle Meta-Learner type to choose correct Explainer
    if isinstance(meta_learner, (cuml.linear_model.LogisticRegression, LogisticRegression)):
        # SHAP LinearExplainer prefers a CPU-based dummy if using cuML
        if isinstance(meta_learner, cuml.linear_model.LogisticRegression):
            explainer_model = LogisticRegression()
            explainer_model.coef_ = meta_learner.coef_.to_numpy()
            explainer_model.intercept_ = meta_learner.intercept_.to_numpy()
        else:
            explainer_model = meta_learner
        
        explainer = shap.LinearExplainer(explainer_model, X_test_meta_df.values)
        shap_values = explainer.shap_values(X_test_meta_df.values)
        
    else:
        # Default for Tree-based models (XGB, LGBM, CatBoost)
        explainer = shap.TreeExplainer(meta_learner)
        shap_values = explainer.shap_values(X_test_meta_df)

    # 2. Plotting logic
    plt.figure(figsize=(10, 6))
    
    # Handle Binary Classification output shapes
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_val_to_plot = shap_values[1]
    else:
        shap_val_to_plot = shap_values

    # Ensure data is on CPU for matplotlib
    if isinstance(shap_val_to_plot, cp.ndarray):
        shap_val_to_plot = cp.asnumpy(shap_val_to_plot)

    shap.summary_plot(shap_val_to_plot, X_test_meta_df, show=False)
    plt.title(f'SHAP Meta-Feature Importance: {prefix}')
    
    # 3. Save
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, f'{prefix}_shap_summary.png')
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP summary saved to {full_path}")