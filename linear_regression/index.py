from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

# load the dataset
dataset = load_diabetes()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["disease_progression"] = dataset.target

# select 'bmi' and 'bp' as predictors
X = df[["bmi", "bp"]]
Y = df["disease_progression"]

# add a constant (needed for statsmodels)
X = sm.add_constant(X)

# create and fit the model
model = sm.OLS(Y, X).fit()

# coefficients of the model
intercept, coef_bmi, coef_bp = model.params

# preprocessing for 3D rendering
x = np.linspace(df["bmi"].min(), df["bmi"].max(), 100)
y = np.linspace(df["bp"].min(), df["bp"].max(), 100)
x, y = np.meshgrid(x, y)
z = intercept + coef_bmi * x + coef_bp * y

# plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d", box_aspect=(2.5, 3, 3))
ax.scatter(
    df["bmi"], df["bp"], df["disease_progression"], c="blue", marker="o", alpha=0.5
)

# fit the plane from regression
ax.plot_surface(x, y, z, color="orange", alpha=0.7)

# labels
ax.set_xlabel("BMI")
ax.set_ylabel("Blood Pressure")
ax.set_zlabel("Disease Progression")

# show chart
plt.show()

# show coefficients
print(model.summary())
