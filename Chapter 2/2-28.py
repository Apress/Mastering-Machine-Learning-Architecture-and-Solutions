df.fillna(df.mean(), inplace=True)  # Impute numerical columns with mean
df.fillna(df.mode().iloc[0], inplace=True)  # Impute categorical columns with mode
