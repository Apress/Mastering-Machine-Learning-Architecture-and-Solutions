import tensorflow as tf

# Load a pre-trained model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Convert the model to TensorFlow Lite (TFLite) format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Save the quantized model
with open("model_quantized.tflite", "wb") as f:
    f.write(quantized_model)
