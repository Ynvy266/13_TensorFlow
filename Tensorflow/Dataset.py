import tensorflow as tf
import numpy as np

# 1. Tạo dữ liệu: 100 số ngẫu nhiên và nhãn tương ứng
features = np.random.rand(100)  # Số thực trong khoảng [0, 1)
labels = (features > 0.5).astype(np.int32)  # 1 nếu > 0.5, ngược lại 0

# 2. Tạo tf.data.Dataset từ hai array
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# 3. Xử lý dataset: shuffle, batch
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(16)

# 4. Lặp qua 2 batch đầu tiên và in ra shape
print("In ra shape của 2 batch đầu:")
for i, (feature_batch, label_batch) in enumerate(dataset.take(2)):
    print(f"Batch {i+1}:")
    print("  Feature shape:", feature_batch.shape)
    print("  Label shape:", label_batch.shape)

