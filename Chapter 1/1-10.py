from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
# Map numerical target to actual species names
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display the first five rows of the dataset
df.head()
