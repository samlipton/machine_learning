"""

Wine identification with random forest

"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
    ("rf", RandomForestClassifier(n_estimators=20, max_depth=2))
])

## fit model
clf.fit(X_tr, y_tr)

## test model
y_pred = clf.predict(X_te)
accuracy = accuracy_score(y_te, y_pred)
print("Accuracy:", accuracy)

## (optimized) parameters
param_grid = {
    "rf__n_estimators": [20, 40, 60],
    "rf__max_depth": [2, 4, 6],
}

grid = GridSearchCV(clf, param_grid, cv=5)
grid.fit(X_tr, y_tr)

## CV results
import json

print("Best parameters:", grid.best_params_)
print("CV params:", json.dumps(grid.cv_results_['params']))
print("CV mean_test_score:", grid.cv_results_['mean_test_score'])
print("CV std_test_score:", grid.cv_results_['std_test_score'])