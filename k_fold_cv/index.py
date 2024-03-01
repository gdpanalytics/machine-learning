from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

# load dataset
dataset = load_diabetes()
X = dataset.data
y = dataset.target

# create model of interest
linear_model = LinearRegression()

# k-fold cross-validation
k = 5  # number of folds
scores = cross_val_score(linear_model, X, y, cv=k, scoring='neg_mean_squared_error')
mse_scores = -scores

# compute mean MSE
mean_mse = np.mean(mse_scores)

print(f"Mean MSE with {k}-folds CV: {mean_mse}")
