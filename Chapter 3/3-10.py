print("\nDefining models for comparison...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Balanced Random Forest": BalancedRandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}
