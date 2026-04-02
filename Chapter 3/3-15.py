import joblib
print("\nSaving the optimized model...")
joblib.dump(optimized_model, 'optimized_model.pkl')
print("Model saved as 'optimized_model.pkl'")
