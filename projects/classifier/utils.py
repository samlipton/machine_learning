"""

Custom functions for Classifiers

"""

import numpy as np
from pathlib import Path

# =============================================================================
# SUtilities
# =============================================================================

def read_column(path: Path, index: int = 0, dtype=float) -> np.ndarray:
    """
    Reads a file into an array
    """

    data = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) > index:
                data.append(parts[index])

    return np.asarray(data, dtype=dtype)


# =============================================================================
# Support vector machines
# =============================================================================

#from sklearn.svm import SVC

# =============================================================================
# Neural networks
# =============================================================================

import tensorflow as tf
from tensorflow import keras

def build_model(n_in=int, n_out=int, n_units=list, l2_reg=1e-3, lr=1e-3):
    """
    Returns a sequential neural network for classifying:
        -n_in    = input size
        -n_out   = output size 
        -n_units = list of hidden units size
    """

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_in,)))

    for n_i in n_units:
        model.add(
            keras.layers.Dense(
                n_i,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ))

    model.add(keras.layers.Dense(n_out, activation="softmax"))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model