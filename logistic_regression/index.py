from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# load the dataset
dataset = load_wine()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["target"] = dataset.target

# separate predictors from target
X = df.drop("target", axis=1)
y = df["target"]

# split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99
)

# create and fit the model on training
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# predict test data
y_pred = model.predict(X_test)

# evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    cmap="Blues",
    fmt="g",
    xticklabels=dataset.target_names,
    yticklabels=dataset.target_names,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
