import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 1. Tải dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Tiền xử lý dữ liệu
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

y_train = keras.utils.to_categorical(y_train, 10)  # one-hot encode
y_test = keras.utils.to_categorical(y_test, 10)

# 3. Mô hình phân loại MNIST bằng Sequential API
model_seq = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation="softmax")  # Phân loại 10 lớp
])

# 4. Compile mô hình
model_seq.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# 5. Huấn luyện mô hình
print("Bắt đầu huấn luyện...")
history = model_seq.fit(x_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.1)
print("Huấn luyện xong!")

# 6. Đánh giá mô hình trên tập test
score = model_seq.evaluate(x_test, y_test, verbose=0)
print("Test Loss (Categorical Crossentropy):", score[0])
print("Test Accuracy:", score[1])

# 7. Biểu đồ loss và accuracy
print("Các thông số được theo dõi trong quá trình huấn luyện:")
print(history.history.keys())

epochs = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(10, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], 'r--', label='Validation Loss')
plt.title('Loss qua các epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'r--', label='Validation Accuracy')
plt.title('Accuracy qua các epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()