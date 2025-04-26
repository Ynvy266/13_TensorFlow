# main.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_cnn_model():
    # Tạo mô hình CNN với cải tiến
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout để tránh overfitting
        layers.Dense(10, activation='softmax')  # Lớp output với 10 lớp cho 10 loại trong CIFAR-10
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def main():
    # Tải dữ liệu CIFAR-10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Tiền xử lý dữ liệu
    train_images, test_images = train_images / 255.0, test_images / 255.0  # Chuẩn hóa ảnh
    train_labels = to_categorical(train_labels, 10)  # One-hot encoding cho nhãn
    test_labels = to_categorical(test_labels, 10)

    # Sử dụng Data Augmentation để tăng cường dữ liệu huấn luyện
    datagen = ImageDataGenerator(
        rotation_range=20,        # Xoay ngẫu nhiên ảnh
        width_shift_range=0.2,    # Dịch chuyển theo chiều ngang
        height_shift_range=0.2,   # Dịch chuyển theo chiều dọc
        shear_range=0.2,          # Biến dạng shear
        zoom_range=0.2,           # Thu phóng ảnh
        horizontal_flip=True,     # Lật ảnh theo chiều ngang
        fill_mode='nearest'       # Điền các vùng trống
    )
    
    # Fit dữ liệu vào ImageDataGenerator
    datagen.fit(train_images)

    # Tạo mô hình CNN
    model = create_cnn_model()

    # 👉 In ra số lượng tham số của mô hình
    print("\n🧠 Thông tin mô hình:")
    model.summary()
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    print(f"\n📊 Tổng tham số: {total_params:,}")
    print(f"📈 Tham số huấn luyện được: {trainable_params:,}")
    print(f"📉 Tham số không huấn luyện được: {non_trainable_params:,}")

    # Huấn luyện mô hình
    print("\n🚀 Bắt đầu huấn luyện mô hình...")

    # Huấn luyện mô hình với Data Augmentation
    model.fit(datagen.flow(train_images, train_labels, batch_size=64),
              epochs=30,
              validation_data=(test_images, test_labels),
              steps_per_epoch=len(train_images) // 64)

    # 👉 Đánh giá mô hình trên tập test
    print("\n📋 Đánh giá mô hình trên tập kiểm tra:")
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
    print(f"✅ Độ chính xác: {accuracy * 100:.2f}%")
    print(f"❌ Độ mất mát (loss): {loss:.4f}")
    
    # Lưu mô hình đã huấn luyện
    model.save('cnn_cifar10_model.h5')
    print("Mô hình đã được lưu thành công!")

if __name__ == "__main__":
    main()