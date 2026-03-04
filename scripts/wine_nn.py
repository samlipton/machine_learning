"""

Wine identification with neural networks

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

## load dataset
wine = datasets.load_wine()
X, y = wine.data, wine.target
# x.shape = (178,13)
# y.shape = (178)
# print(wine.feature_names,wine.target_names)

## build model 
def build_model(n_units=int):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(13,)),
        keras.layers.Dense(
            n_units, 
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-3)),
        keras.layers.Dense(3, activation="softmax")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

## transform dataset
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

## train model
def train_model(n_units=8):
    loss_sc,acc_sc = [],[]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X,y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit scaler ONLY on training fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = build_model(n_units)

        history = model.fit(
            X_train, y_train,
            epochs=80,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True)
        ])

        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        #print(f"\nFold {fold + 1}:")
        #print(f"Accuracy: {acc:.3f}")
        #print(f"Loss:     {loss:.3f}")
        loss_sc.append(loss); acc_sc.append(acc)

    return model,loss_sc,acc_sc

## results
n_units = 8
model,loss,acc = train_model(n_units)
# print(model._layers[1].weights[1].shape.as_list()[0])
print(f"\nCross-validation results for n={n_units}")
print(f"Accuracy: {np.mean(acc):.3f} ± {np.std(acc):.3f}")
print(f"Loss:     {np.mean(loss):.3f} ± {np.std(loss):.3f}")