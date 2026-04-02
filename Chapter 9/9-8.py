mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
feature_importance = dict(zip(iris.feature_names, mean_abs_shap))
print("Mean Absolute SHAP Values (Feature Importance):")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.3f}")

# In practice, these values can be compared with domain knowledge to ensure fair model behavior.
