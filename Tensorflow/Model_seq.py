import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 1. Mô hình hồi quy bằng Sequential API đã xây dựng ở Chương 3
model_seq = keras.Sequential([
    keras.Input(shape=(10,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation="linear")
])

# 2. Tạo dữ liệu huấn luyện và kiểm thử giả: Y = 2 * X + noise
np.random.seed(0)
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
noise = np.random.normal(0, 0.1, size=(1000, 1))
Y = 2 * X.sum(axis=1, keepdims=True) + noise  # Y = 2 * tổng các đặc trưng + nhiễu

# Chia dữ liệu thành tập train/test
x_train, x_test = X[:800], X[800:]
y_train, y_test = Y[:800], Y[800:]

# 3. Compile mô hình
model_seq.compile(optimizer='adam',
              loss='mse',       # Mean Squared Error
              metrics=['mae'])  # Mean Absolute Error

# 4. Huấn luyện mô hình
print("Bắt đầu huấn luyện...")
history = model_seq.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1)
print("Huấn luyện xong!")

# 5. Đánh giá mô hình trên tập test
score = model_seq.evaluate(x_test, y_test, verbose=0)
print("Test Loss (MSE):", score[0])
print("Test MAE:", score[1])

# Xem lịch sử huấn luyện
print("Các thông số được theo dõi trong quá trình huấn luyện:")
print(history.history.keys())

# Lấy số epoch
epochs = range(1, len(history.history['loss']) + 1)

# Biểu đồ Loss (MSE)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], 'r--', label='Validation Loss')
plt.title('Loss (MSE) qua các epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# Biểu đồ MAE
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['mae'], 'b-', label='Training MAE')
plt.plot(epochs, history.history['val_mae'], 'r--', label='Validation MAE')
plt.title('MAE qua các epoch')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()