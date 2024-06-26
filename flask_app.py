from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
mphands = mp.solutions.hands
handprocessor = mphands.Hands(static_image_mode=True)

# STEP 2: Create an GestureRecognizer object.
model_file = open('./models/gesture_recognizer.task', "rb")
model_data = model_file.read()
model_file.close()

base_options = python.BaseOptions(model_asset_buffer=model_data)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)


app = Flask(__name__)

def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

# Dummy function for prediction, replace with actual model prediction logic
def predict_hand_sign(image_array):
    # Replace this with the actual prediction logic using your trained model
    print(image_array.shape)
    processed_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    solution = recognizer.recognize(processed_image)
    if solution.gestures[0][0]:
        print(solution.gestures[0][0].category_name)
        return solution.gestures[0][0].category_name
    else: return("No gesture recognized.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Assuming the image is base64 encoded
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess the image as needed for your model
        #image = image.resize((64, 64))  # Example resize
        image_array = np.array(image)  # Convert to numpy array
        #image_array = image_array / 255.0  # Example normalization
        
        # Get the prediction
        prediction = predict_hand_sign(image_array)
        
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
