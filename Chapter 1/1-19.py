def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Creating a DataFrame for the input
    input_data = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })

    # Scaling the input data


input_scaled = scaler.transform(input_data)

# Making prediction`
prediction = rf_classifier.predict(input_scaled)

# Mapping numerical prediction to species name
species = le.inverse_transform(prediction)[0]
return species

# Example usage of the prediction function
example_species = predict_species(5.1, 3.5, 1.4, 0.2)
print(f"The predicted species is: {example_species}")
