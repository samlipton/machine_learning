"""

Wine identification with neural networks

"""

import tensorflow as tf
from tensorflow import keras

## load dataset
from sklearn import datasets

wine = datasets.load_wine()
x,y = wine.data,wine.target
# x.shape = (178,13)
# y.shape = (178)
print(wine.feature_names,wine.target_names)

## transform dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([('std_scaler', StandardScaler())])
X = pipeline.fit_transform(x)

## split dataset
# add k-fold cross-validation
n0,n1,n2,n3 = 0,128,143,178
x_tr,y_tr = X[n0:n1],y[n0:n1]
x_va,y_va = X[n1:n2],y[n1:n2]
x_te,y_te = X[n2:n3],y[n2:n3]

## build model
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(shape=[13]))
model.add(keras.layers.Dense(20, activation="relu"))
model.add(keras.layers.Dense(10, activation="relu"))
model.add(keras.layers.Dense(3, activation="softmax"))
print(model.summary())

## compile model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

## train model
history = model.fit(x_tr,y_tr,epochs=30,validation_data=(x_va,y_va))

print(y_va)

