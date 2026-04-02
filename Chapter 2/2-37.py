# Feature Extraction: Create a new feature 'FamilySize' from 'SibSp' and 'Parch'
data_cleaned['FamilySize'] = data_cleaned['SibSp'] + data_cleaned['Parch'] + 1

# Feature Transformation: Convert 'Sex' into binary values (0 for male, 1 for female)
data_cleaned['Sex'] = data_cleaned['Sex'].map({'male': 0, 'female': 1})

# Feature Creation: Create a new feature 'IsAlone' based on 'FamilySize'
data_cleaned['IsAlone'] = (data_cleaned['FamilySize'] == 1).astype(int)

# Feature Transformation: Binning 'Age' into categories (e.g., Child, Teen, Adult, Senior)
bins = [0, 12, 18, 60, 100]
labels = ['Child', 'Teen', 'Adult', 'Senior']
data_cleaned['AgeGroup'] = pd.cut(data_cleaned['Age'], bins=bins, labels=labels)

# Display the updated dataset
print("\nDataset After Feature Engineering:")
print(data_cleaned.head())

# Visualize the new 'AgeGroup' feature with respect to survival
plt.figure(figsize=(8, 5))
sns.countplot(x='AgeGroup', hue='Survived', data=data_cleaned)
plt.title('Survival Rate by Age Group')
plt.show()
