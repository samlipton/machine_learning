"""

Wine identification with neural networks

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

wine = datasets.load_wine()
X, y = wine.data, wine.target

def build_model(n_units, l2_reg, lr):
    model = keras.Sequential([
        keras.layers.Input(shape=(13,)),
        keras.layers.Dense(
            n_units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2_reg)),
        keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

param_grid = {
    "n_units": [6, 8, 12],
    "l2_reg": [1e-4, 1e-3, 1e-2],
    "lr": [1e-4, 3e-4, 1e-3]
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

best_score = 0
best_params = None

for n_units in param_grid["n_units"]:
    for l2_reg in param_grid["l2_reg"]:
        for lr in param_grid["lr"]:
            scores = []

            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                model = build_model(n_units, l2_reg, lr)

                model.fit(
                    X_train, y_train,
                    epochs=100,
                    verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            patience=10,
                            restore_best_weights=True
                )])

                _, acc = model.evaluate(X_val, y_val, verbose=0)
                scores.append(acc)

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = (n_units, l2_reg, lr)

print("Best params:", best_params)
print("Best CV accuracy:", best_score)