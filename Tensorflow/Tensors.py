import tensorflow as tf

# Tạo tensor 2D (ma trận) 3x3 với giá trị ngẫu nhiên trong khoảng [0, 1)
tensor_2d = tf.random.uniform(shape=(3, 3), minval=0, maxval=1)
print("Tensor 2D ban đầu:\n", tensor_2d.numpy())

# Tạo tensor 1D với 3 phần tử
tensor_1d = tf.constant([1, 2, 3])
print("Tensor 1D:\n", tensor_1d.numpy())

# Nhân với ma trận đơn vị (giá trị giữ nguyên)
identity = tf.eye(3)
multiplied = tf.matmul(tensor_2d, identity)
print("Kết quả nhân với ma trận đơn vị:\n", multiplied.numpy())

# Tính tổng các phần tử trong tensor
sum_elements = tf.reduce_sum(tensor_2d)
print("Tổng các phần tử trong tensor_2d:", sum_elements.numpy())

# Thay đổi phần tử [0, 0] thành 99 bằng tf.Variable
tensor_var = tf.Variable(tensor_2d)
tensor_var[0, 0].assign(99)
print("Tensor sau khi thay đổi phần tử [0,0]:\n", tensor_var.numpy())