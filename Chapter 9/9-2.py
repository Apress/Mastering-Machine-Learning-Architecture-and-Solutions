import shap
import xgboost
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load dataset and train model
california = fetch_california_housing(as_frame=True)
X = california.data
y = california.target

# Train model
model = xgboost.XGBRegressor().fit(X, y)

# Create a SHAP explainer
explainer = shap.Explainer(model, X)

# Calculate SHAP values for a one-row DataFrame.
shap_values = explainer(X.iloc[0:1])
# Visualize the explanation for the first (and only) instance.
shap.plots.waterfall(shap_values[0])
