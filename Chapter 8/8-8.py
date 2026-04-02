from sklearn.metrics import accuracy_score

# Sample predictions for two demographic groups
group_A_actual = [1, 0, 1, 1, 0, 1]
group_A_pred = [1, 0, 1, 1, 1, 1]

group_B_actual = [1, 0, 1, 0, 0, 0]
group_B_pred = [1, 1, 1, 0, 0, 0]

# Calculate accuracy for each group
accuracy_A = accuracy_score(group_A_actual, group_A_pred)
accuracy_B = accuracy_score(group_B_actual, group_B_pred)

print(f"Accuracy for Group A: {accuracy_A:.2f}")
print(f"Accuracy for Group B: {accuracy_B:.2f}")
