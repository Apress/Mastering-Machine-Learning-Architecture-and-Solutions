plt.figure(figsize=(12, 8))
for idx, column in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, idx+1)
    sns.histplot(data=df, x=column, kde=True, bins=20)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()
