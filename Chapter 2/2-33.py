train_mean = train_data.mean()
test_mean = test_data.mean()

diff_percentage = ((train_mean - test_mean) / train_mean) * 100
print("Feature distribution differences:\n", diff_percentage)
