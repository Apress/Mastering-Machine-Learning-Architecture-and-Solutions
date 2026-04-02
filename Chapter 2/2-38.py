# Separate features and target variable
X = data_cleaned.drop(columns=['Survived'])
y = data_cleaned['Survived']

# Define numerical and categorical columns for preprocessing
numerical_cols = ['Age', 'Fare', 'FamilySize']
categorical_cols = ['Sex', 'Embarked', 'Pclass', 'AgeGroup']

# Preprocessing for numerical data: Impute missing values and scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: Impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)
