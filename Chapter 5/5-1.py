import tensorflow as tf
import time

# Create a simple matrix multiplication workload
A = tf.random.normal([1000, 1000])
B = tf.random.normal([1000, 1000])

# Run on CPU
with tf.device('/CPU:0'):
    start_time = time.time()
    result_cpu = tf.matmul(A, B)
    print("CPU Execution Time:", time.time() - start_time)

# Run on GPU
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        start_time = time.time()
        result_gpu = tf.matmul(A, B)
        print("GPU Execution Time:", time.time() - start_time)
else:
    print("No GPU detected.")
