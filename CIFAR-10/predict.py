# predict.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model

def load_and_preprocess_image(image_path):
    # Tải ảnh và chuẩn hóa
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
    img_array = img_array / 255.0  # Chuẩn hóa ảnh
    return img_array

def predict_image(model, image_path):
    # Dự đoán ảnh
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)  # Lấy chỉ số lớp có xác suất cao nhất
    return class_idx, predictions[0][class_idx]

def main(image_path):
    # Tải mô hình đã huấn luyện
    model = load_model('vgg16_cifar10_model_old.h5')

    # Dự đoán ảnh
    class_idx, probability = predict_image(model, image_path)

    # Tải tên các lớp trong CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Hiển thị kết quả
    print(f"Ảnh được dự đoán là: {class_names[class_idx]} với xác suất: {probability * 100:.2f}%")

    # Hiển thị ảnh
    img = tf.keras.preprocessing.image.load_img(image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[class_idx]} ({probability*100:.2f}%)")
    plt.show()

if __name__ == "__main__":
    image_path = 'true_cat_s_000619.png'
    main(image_path)