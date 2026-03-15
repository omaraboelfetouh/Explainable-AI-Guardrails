import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def run_shap_analysis():
    print("Generating synthetic data...")
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    feature_names = [f"Feature_{i}" for i in range(10)]
    
    print("Training RandomForest Classifier...")
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a subset of data
    sample_X = X[:50]
    shap_values = explainer.shap_values(sample_X)
    
    print("SHAP values computed successfully.")
    print(f"Shape of SHAP values: {np.array(shap_values).shape}")
    print("Use shap.summary_plot(shap_values, sample_X) to visualize.")

if __name__ == "__main__":
    run_shap_analysis()