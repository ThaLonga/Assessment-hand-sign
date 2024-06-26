# Assessment-hand-sign
This hand-sign-prediction repository contains an app "app.py" which runs using the camera and detects gestures using Mediapipe. Further, it also recognizes waving motion by employment of a LSTM-based neural network. The Flask web app "app_flask.py" accepts jpeg image uploads and returns the recognized gesture.

## Requirements
The code was built using python 3.11, flask 3.0, tensorflow 2.16, keras 3.3.3, mediapipe 0.10 and openCV 4.10. Further details can be found in requirements.txt.

## Technology Stack
Backend: Flask, Python, MediaPipe
Frontend: HTML, CSS, JavaScript
Model: TensorFlow, MediaPipe

## Project structure:
│\
├── app.py                  # Main application file\
├── app_flask.py            # Web application file\
├── waving_training.ipynb   # Notebook for training waving detector\
├── benchmark.ipynb         # Mediapipe GestureRecognizer performance benchmark\
├── templates/\
│   └── index.html          # HTML file for the web interface\
├── models/\
│   └── gesture_recognizer.task  # Trained model file for gesture recognition\
│   └── waving.keras        # Trained model file for waving detection\
├── requirements.txt        # List of Python packages required for the project\
└── README.md
