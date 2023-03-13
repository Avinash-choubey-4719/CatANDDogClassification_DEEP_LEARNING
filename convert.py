import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('model1_catsVSdogs_10epoch.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
