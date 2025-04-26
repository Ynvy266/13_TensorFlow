import tensorflow as tf
import numpy as np
import datetime  # Để tạo thư mục log duy nhất
from tensorflow import keras
from tensorflow.keras import layers

# 1. Mô hình hồi quy
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

# 4. Tạo thư mục log duy nhất cho TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 5. Cấu hình callback ModelCheckpoint
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='models/best_regression_model.keras',  # Lưu mô hình tốt nhất
    save_weights_only=False,  # Lưu cả mô hình chứ không chỉ trọng số
    monitor='val_loss',       # Theo dõi validation loss
    mode='min',               # Lưu khi validation loss thấp nhất
    save_best_only=True)      # Chỉ lưu mô hình tốt nhất

# 6. Cấu hình callback EarlyStopping
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',         # Theo dõi validation loss
    patience=5,                 # Dừng nếu không cải thiện sau 5 epochs
    restore_best_weights=True)  # Khôi phục trọng số tốt nhất khi dừng

# 7. Huấn luyện mô hình với Callbacks
print("Bắt đầu huấn luyện với Callbacks và TensorBoard...")
history = model_seq.fit(x_train, y_train,
                        epochs=20,
                        batch_size=32,
                        validation_split=0.1,
                        callbacks=[tensorboard_callback,
                                   model_checkpoint_callback,
                                   early_stopping_callback])

print("Huấn luyện xong!")

# 8. Đánh giá mô hình
score = model_seq.evaluate(x_test, y_test, verbose=0)
print("Test Loss (MSE):", score[0])
print("Test MAE:", score[1])

# 9. Lưu mô hình cuối cùng
model_seq.save('models/final_regression_model.keras')

# 10. Tải lại mô hình tốt nhất
best_model = keras.models.load_model('models/best_regression_model.keras')
score_best_model = best_model.evaluate(x_test, y_test, verbose=0)
print("Best Model Test Loss (MSE):", score_best_model[0])
print("Best Model Test MAE:", score_best_model[1])

# 11. Hướng dẫn chạy TensorBoard
# Mở terminal
# Chạy file .py
# Chạy lệnh: tensorboard --logdir=logs/fit
# Truy cập vào đường dẫn: http://localhost:6006/