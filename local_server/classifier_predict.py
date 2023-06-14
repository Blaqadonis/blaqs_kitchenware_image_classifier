import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify

# Load the trained model
model = keras.models.load_model('xception_v4_large_12_0.965.h5')

# Define the labels
labels = {
    0: 'cup',
    1: 'fork',
    2: 'glass',
    3: 'knife',
    4: 'plate',
    5: 'spoon'
}

# Process the image
def process_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    image_size = (299, 299)  # Set your desired image size
    img = img.resize(image_size)
    x = img_to_array(img)
    X = np.array([x])
    X = preprocess_input(X)
    return X

# Make predictions
def predict(image):
    preds = model.predict(image)
    answer = labels[preds[0].argmax()]
    return answer

# Flask app
app = Flask('Blaqs_Kitchenware_Classifier')

@app.route('/classifier_predict', methods=['POST'])
def predict_endpoint():
    url = request.form['url']
    image = process_image(url)
    pred = predict(image)

    result = {
        'Prediction': f'This is {pred}'
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
