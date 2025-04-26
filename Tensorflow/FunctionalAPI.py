import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model

inputs = Input(shape=(10,))
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(10, activation="linear", name="output")(x)

model_func = Model(inputs=inputs, outputs=outputs, name="functional_regression_model")
model_func.summary()