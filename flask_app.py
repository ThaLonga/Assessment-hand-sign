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

def predict_hand_sign(image_array):
    processed_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    solution = recognizer.recognize(processed_image)
    if solution.gestures:
        return solution.gestures[0][0].category_name
    else: return("No hand recognized.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)  # Convert to numpy array
        
        # Get the prediction
        prediction = predict_hand_sign(image_array)
        
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
