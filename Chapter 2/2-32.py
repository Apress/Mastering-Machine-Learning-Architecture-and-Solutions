# Check for duplicate rows
duplicate_rows = df[df.duplicated()]
print("Number of duplicate records:", duplicate_rows.shape[0])
