import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular

# Load dataset and train model
iris = load_iris()
X, y = iris.data, iris.target
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

# Create a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)

# Explain a prediction
i = 25  # Index of the instance to explain
exp = explainer.explain_instance(X[i], rf.predict_proba, num_features=2)
exp.show_in_notebook()

