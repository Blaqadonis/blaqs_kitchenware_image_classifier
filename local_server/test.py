import requests

# URL of the Flask app
url = 'http://localhost:9696/classifier_predict'

# image URL for prediction
image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_LMmuWRP3JXQEHdfRkF0Gkaa4MofCs6MbXMcV__I&s'

# Create a dictionary with the image URL
data = {
    'url': image_url
}

# Send a POST request to the Flask app
response = requests.post(url, data=data)

# Check the response status code
if response.status_code == 200:
    # Retrieve the prediction result
    result = response.json()
    #prediction = result['prediction']
    print(result)
else:
    print('Error occurred during prediction.')
