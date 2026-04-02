# Create a SHAP TreeExplainer using the trained decision tree
explainer = shap.TreeExplainer(clf)
# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Plot a summary plot of SHAP values for the positive class (class 0)
explainer = shap.Explainer(clf, X_train)
shap_values = explainer(X_test)

# Now, shap_values.values should have shape (n_samples, n_features)
print("SHAP values shape:", shap_values.values.shape)  # Expect (30, 4)

# Plot the summary plot using the new SHAP values
shap.summary_plot(shap_values.values, X_test, feature_names=np.array(iris.feature_names))
