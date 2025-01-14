import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from handgest import HandGestureRecognition

# Dimensiunile ecranului
screen_width, screen_height = pyautogui.size()

# Inițializări MediaPipe și model de recunoaștere
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Recunoașterea gesturilor
recognizer = HandGestureRecognition(model_path='hand_recognition.tflite')

# Filtru pentru stabilizarea gesturilor
gesture_history = []
GESTURE_FILTER_SIZE = 5

def stable_gesture(gesture_index):
    """Stabilizează detectarea gesturilor pe baza istoriei."""
    gesture_history.append(gesture_index)
    if len(gesture_history) > GESTURE_FILTER_SIZE:
        gesture_history.pop(0)
    # Returnează gestul cel mai frecvent
    return max(set(gesture_history), key=gesture_history.count)

# Funcție pentru mișcarea cursorului
def move_cursor(landmarks):
    x = int(landmarks[4][0] * screen_width)
    y = int(landmarks[4][1] * screen_height)
    mirrored_x = screen_width - x
    mirrored_y = screen_height - y
    pyautogui.moveTo(mirrored_x, mirrored_y, duration=0.1)  # Smooth transition

# Funcție pentru detectarea gestului
def handle_gesture(gesture_index, landmarks):
    stable_index = stable_gesture(gesture_index)
    if stable_index == 0:  # Palma deschisă
        move_cursor(landmarks)
    elif stable_index == 1:  # Pumn închis
        pyautogui.click()
    elif stable_index == 2:  # Deget arătător ridicat
        pyautogui.rightClick()

# Funcție pentru preprocesarea landmarks
def preprocess_landmarks(landmarks):
    flat_landmarks = np.array(landmarks, dtype=np.float32).flatten()
    return np.expand_dims(flat_landmarks, axis=0)

# Deschide camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            input_data = preprocess_landmarks(landmarks_list)
            gesture_index = recognizer(input_data)
            handle_gesture(gesture_index, landmarks_list)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f'Gesture: {gesture_index}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
