"""

Wine identification with neural networks

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn import datasets
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scikeras.wrappers import KerasClassifier

## load data
wine = datasets.load_wine()
X, y = wine.data, wine.target

## build model 
def build_model(n_units=8, l2_reg=1e-3, lr=3e-4):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(13,)),
        keras.layers.Dense(
            n_units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2_reg)        ),
        keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

## wrap model
clf = KerasClassifier(
    model=build_model,
    epochs=100,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
)])

## pipeline 
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", clf)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

## parameter grid
param_grid = {
    "clf__model__n_units": [6, 8, 12],
    "clf__model__l2_reg": [1e-4, 1e-3, 1e-2],
    "clf__model__lr": [1e-4, 3e-4, 1e-3],
}

## grid search
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X, y)

print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)
