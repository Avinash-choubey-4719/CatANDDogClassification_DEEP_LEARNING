from flask import Flask
from flask import request
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # code to make predictions with your deep learning model

    data = request.get_json()

    model = tf.keras.models.load_model('model1_catsVSdogs_10epoch.h5')

    x = preprocess_input(data)

    predictions = model.predict(x)

    return {'predicions' : predictions.tolist()}

if __name__ == "__main__":
        app.run(debug=True, host = '0.0.0.0', port = 5000)
