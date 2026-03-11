"""

Wine identification with support vector machine

"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

## load dataset
wine = datasets.load_wine()
X, y = wine.data, wine.target

# train-test split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, 
                                          stratify=y, random_state=42)

## build model
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

## fit model
clf.fit(X_tr, y_tr)

## test model
y_pred = clf.predict(X_te)
accuracy = accuracy_score(y_te, y_pred)
print("Accuracy:", accuracy)

## (optimized) parameters
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": [1, 0.1, 0.01, 0.001],
    "svm__kernel": ["rbf"]
}

grid = GridSearchCV(clf, param_grid, cv=5)
grid.fit(X_tr, y_tr)

print("Best parameters:", grid.best_params_)