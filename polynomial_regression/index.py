from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
dataset = load_diabetes()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['disease_progression'] = dataset.target

# select 'bmi' and 'bp' as predictors
X = df[['bmi', 'bp']]
Y = df['disease_progression']

# trasform features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# add a constant (needed for statsmodels)
X_poly = sm.add_constant(X_poly)

# create and fit the model
model = sm.OLS(Y, X_poly).fit()

# create a grid for 'bmi' and 'bp'
bmi = np.linspace(df['bmi'].min(), df['bmi'].max(), 100)
bp = np.linspace(df['bp'].min(), df['bp'].max(), 100)
xx, yy = np.meshgrid(bmi, bp)
zz = model.predict(sm.add_constant(poly.transform(np.c_[xx.ravel(), yy.ravel()])))
zz = zz.reshape(xx.shape)

# plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d', box_aspect=(2.5,3,3))
ax.scatter(df['bmi'], df['bp'], df['disease_progression'], marker='o', alpha=0.5)
ax.plot_surface(xx, yy, zz, color='orange', alpha=0.7)

# formatting
ax.set_xlabel('BMI')
ax.set_ylabel('Blood Pressure')
ax.set_zlabel('Disease Progression')

# show chart
plt.show()
