import tensorflow as tf

# Detect and assign TPU if available
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
except ValueError:
    strategy = tf.distribute.MirroredStrategy()  # Fall back to GPU/CPU if TPU unavailable

print("Using strategy:", strategy)
