plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=["Not Class 0", "Class 0"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
