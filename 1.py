import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("mask_detection_model.h5")

# Define a function that takes a float32 input of shape [None, 128, 128, 3]
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 128, 128, 3], dtype=tf.float32)])
def model_func(x):
    return model(x)

# Convert to ConcreteFunction
concrete_func = model_func.get_concrete_function()

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the TFLite model
with open("mask_detection_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted to TFLite successfully.")
