print("\nLoading the dataset...")
data = pd.read_csv('creditcard.csv')

# Display basic information about the dataset
print("\nDataset Overview:")
print(data.info())
print("\nClass Distribution:")
print(data['Class'].value_counts())
