print("\nHandling imbalanced data using multiple techniques...")

# Split the data into training and testing sets
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Technique 1: SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Technique 2: Undersampling (Random Undersampling)
X_train_undersampled, y_train_undersampled = resample(
    X_train[y_train == 0], y_train[y_train == 0],
    replace=False, n_samples=len(y_train[y_train == 1]), random_state=42
)
X_train_undersampled = pd.concat([X_train_undersampled, X_train[y_train == 1]])
y_train_undersampled = pd.concat([y_train_undersampled, y_train[y_train == 1]])

# Technique 3: Class Weighting (for models that support it)
class_weights = {0: 1, 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}
