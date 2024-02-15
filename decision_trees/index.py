from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# load dataset
dataset = load_diabetes()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
X = df[['bmi', 'bp']]
y = dataset.target

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# create and fit model
model = DecisionTreeRegressor(max_depth=2, random_state=99)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse_test}")

# visualize decision tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=['bmi', 'bp'], filled=True)
plt.show()