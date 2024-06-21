import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
VisionRunningMode = mp.tasks.vision.RunningMode


# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='./models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=VisionRunningMode.IMAGE)
recognizer = vision.GestureRecognizer.create_from_options(options)

def main():
    cap = cv2.VideoCapture(0)
    hands = mphands.Hands()
    while True:
        data, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        recognition_result = recognizer.recognize(mp_image)
        brect = None
        if results.multi_hand_landmarks:
            brect = calc_bounding_rect(image, results.multi_hand_landmarks[0])
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks, mphands.HAND_CONNECTIONS
                )

        #if recognition_result:
        #    top_gesture = recognition_result.gestures[0][0]
        #    hand_landmarks = recognition_result.hand_landmarks

        if brect:
            image = draw_bounding_rect(True, image, brect=brect)
            if recognition_result:
                image = draw_info_text(image=image, brect=brect, recognition_result=recognition_result)




        cv2.imshow('Handtracker', image)
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

    # After the loop release the cap object 
    cap.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, recognition_result):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if len(recognition_result.gestures)>0:
        info_text = recognition_result.gestures[0][0].category_name
        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image

if __name__ == '__main__':
    main()