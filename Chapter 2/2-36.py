# Check for missing values before cleaning
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# Drop irrelevant columns (e.g., 'Name', 'Ticket', 'Cabin')
# These columns are either too specific or have too many missing values
data_cleaned = data.drop(columns=['Name', 'Ticket', 'Cabin'], errors='ignore')

# Impute missing values for numerical columns (e.g., 'Age') with the median
data_cleaned['Age'].fillna(data_cleaned['Age'].median(), inplace=True)

# Impute missing values for categorical columns (e.g., 'Embarked') with the mode
data_cleaned['Embarked'].fillna(data_cleaned['Embarked'].mode()[0], inplace=True)

# Drop rows with missing values in the target column (if any)
data_cleaned.dropna(subset=['Survived'], inplace=True)

# Verify cleaning results
print("\nMissing Values After Cleaning:")
print(data_cleaned.isnull().sum())

# Check for duplicate rows and remove them if any
print(f"\nNumber of Duplicate Rows Before Removal: {data_cleaned.duplicated().sum()}")
data_cleaned.drop_duplicates(inplace=True)
print(f"Number of Duplicate Rows After Removal: {data_cleaned.duplicated().sum()}")

# Visualize the distribution of the target variable ('Survived')
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=data_cleaned)
plt.title('Distribution of Survived')
plt.show()
