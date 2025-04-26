import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Mô hình hồi quy đơn giản với đầu vào (10,)
model_seq = keras.Sequential(
    [
        keras.Input(shape=(10,)),                 # Input vector 10 chiều
        layers.Dense(64, activation="relu"),      # Lớp ẩn thứ 1
        layers.Dense(32, activation="relu"),      # Lớp ẩn thứ 2
        layers.Dense(10, activation="linear")
    ],
    name="sequential_regression_model"
)

model_seq.summary()