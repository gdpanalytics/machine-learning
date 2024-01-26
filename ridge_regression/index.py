from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# load the dataset
dataset = load_diabetes()
X = dataset.data
y = dataset.target

# split the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99
)

# generate some positive lambda values
lambdas = np.logspace(-4, 4, 200)

# save coefficients estimate and MSE
coefs = []
mse = []

# for each lambda, fit the model and compute MSE
for lmb in lambdas:
    ridge = Ridge(alpha=lmb)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)
    y_pred = ridge.predict(X_test)
    mse.append(mean_squared_error(y_test, y_pred))

# find the lambda corresponding to the lowest MSE
min_mse = min(mse)
best_alpha = lambdas[mse.index(min_mse)]

# plot coefficients to see shrinkage
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(lambdas, coefs)
ax.set_xscale("log")
plt.xlabel("Lambda (log scale)")
plt.ylabel("Coefficients")
plt.show()

# print best lambda and corresponding MSE
print(best_alpha, min_mse)
