import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Download the image from the internet
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_LMmuWRP3JXQEHdfRkF0Gkaa4MofCs6MbXMcV__I&s'
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Preprocess the image
image_size = (299, 299)  # Set your desired image size
img = img.resize(image_size)
x = img_to_array(img)
X = np.array([x])
X = preprocess_input(X)

model = keras.models.load_model('xception_v4_large_12_0.965.h5')
labels = {
    0: 'cup',
    1: 'fork',
    2: 'glass',
    3: 'knife',
    4: 'plate',
    5: 'spoon'
}

# Make predictions
pred = model.predict(X)
answer = labels[pred[0].argmax()]
print(answer)