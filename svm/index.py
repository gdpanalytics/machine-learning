from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# load dataset
wine_dataset = load_wine()
X = wine_dataset.data
y = wine_dataset.target

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99
)

# create and fit model
model = SVC(kernel="rbf", C=1.0, gamma="auto")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza sui dati di test: {accuracy}")
