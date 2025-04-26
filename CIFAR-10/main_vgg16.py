# VGG-16 cho CIFAR-10

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16

def create_vgg16_model():
    # Tải mô hình VGG-16 với trọng số ImageNet và loại bỏ phần fully connected (include_top=False)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Đóng băng các lớp của VGG-16 để không huấn luyện lại chúng
    base_model.trainable = False

    # Tạo mô hình hoàn chỉnh bằng cách thêm các lớp tùy chỉnh
    model = models.Sequential([
        base_model,
        layers.Flatten(),  # Dùng Flatten thay vì GlobalAveragePooling2D
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Dropout để giảm overfitting
        layers.Dense(10, activation='softmax')  # 10 lớp tương ứng với các nhãn CIFAR-10
    ])

    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def main():
    # Tải dữ liệu CIFAR-10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # Tạo mô hình VGG-16
    model = create_vgg16_model()

    # Hiển thị thông tin mô hình
    model.summary()

    # Huấn luyện mô hình
    model.fit(train_images, train_labels, batch_size=64, epochs=50, validation_data=(test_images, test_labels))

    # Đánh giá mô hình
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Độ chính xác trên tập test: {test_acc * 100:.2f}%")

    # Lưu mô hình
    model.save("vgg16_cifar10_model.h5")
    print("Mô hình đã được lưu.")

if __name__ == "__main__":
    main()