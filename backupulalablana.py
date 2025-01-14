import csv
import copy
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui
from handgest import HandGestureRecognition
import time

# Disable PyAutoGUI fail-safe for this example (not recommended for production)
pyautogui.FAILSAFE = False

# Variables for cursor movement stability
previous_x, previous_y = None, None
cursor_smooth_factor = 0.2  # 0.2 will smooth out the cursor movements

# Variables for click detection
last_click_time = 0  # to manage debounce for clicks

def main():
    global previous_x, previous_y, last_click_time
    # Pregătirea capturii video ------------------------------------------
    cap_device = 0  
    cap_width = 960  
    cap_height = 540 

    use_static_image_mode = False  
    min_detection_confidence = 0.7  
    min_tracking_confidence = 0.5  
    use_brect = True  

    # Pregătirea camerei ---------------------------------
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Incarcarea modelului mediapipe pentru Hand Recognition ------------------------------
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Definim un obiect de tipul HandGestureRecognition
    hand_gesture_recognition = HandGestureRecognition()

    # Citirea etichetelor din fisierul de etichete (labels) ---------------------------
    with open('labels.csv', encoding='utf-8-sig') as f:
        labels = csv.reader(f)
        labels = [row[0] for row in labels]

    mode = 0

    while True:
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        
        number, mode = select_mode(key, mode)

        # Capturăm imaginea de la cameră
        ret, image = cap.read()
        if not ret:
            break

        # Inversăm imaginea orizontal
        image = cv.flip(image, 1)

        # Creăm o copie a imaginii pentru a lucra pe aceasta
        debug_image = copy.deepcopy(image)

        # Aplicăm modelul de detectare a mâinii
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Preprocesăm lista de landmark-uri
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Logăm datele
                logging_csv(number, mode, pre_processed_landmark_list)

                # Recunoaștem gestul
                hand_sign_id = hand_gesture_recognition(pre_processed_landmark_list)

                # Mișcăm cursorul pentru gestul 0
                if hand_sign_id == 0:
                    move_cursor_from_landmarks(landmark_list)

                # Detectăm click stânga sau dreapta
                if hand_sign_id == 1:  # Dacă gestul este pentru click stânga (gestul 1)
                    if time.time() - last_click_time > 0.5:  # Debounce pentru click
                        pyautogui.click(button='left')
                        last_click_time = time.time()

                elif hand_sign_id == 2:  # Dacă gestul este pentru click dreapta (gestul 2)
                    if time.time() - last_click_time > 0.5:  # Debounce pentru click
                        pyautogui.click(button='right')
                        last_click_time = time.time()

                # Desenăm informațiile pe imagine
                if 0 <= hand_sign_id < len(labels):  # Verifică dacă hand_sign_id este valid
                    debug_image = draw_info_text(debug_image, brect, handedness, labels[hand_sign_id])
                else:
                    debug_image = draw_info_text(debug_image, brect, handedness, "Unknown")

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

        debug_image = draw_info(debug_image, mode, number)

        # Afișăm imaginea
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def move_cursor(x, y):
    global previous_x, previous_y
    screen_width, screen_height = pyautogui.size()  # Dimensiunile ecranului
    # Asigură-te că coordonatele sunt în intervalul [0, 1]
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    # Proiectăm coordonatele relative pe dimensiunile ecranului
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)

    # Stabilizarea mișcării cursorului (smooth movement)
    if previous_x is None or previous_y is None:
        previous_x, previous_y = screen_x, screen_y
    screen_x = int(previous_x + cursor_smooth_factor * (screen_x - previous_x))
    screen_y = int(previous_y + cursor_smooth_factor * (screen_y - previous_y))

    # Asigură-te că mișcarea cursorului nu depășește limitele ecranului
    screen_x = np.clip(screen_x, 0, screen_width - 1)
    screen_y = np.clip(screen_y, 0, screen_height - 1)

    pyautogui.moveTo(screen_x, screen_y)  # Mișcăm cursorul pe coordonatele respective
    previous_x, previous_y = screen_x, screen_y

def move_cursor_from_landmarks(landmark_list):
    """
    Această funcție mapază coordonatele degetului mare (thumb tip) pentru mișcarea cursorului.
    """
    try:
        # Landmark-ul pentru vârful degetului mare (thumb tip, index 4)
        thumb_tip = landmark_list[4]
        thumb_x = thumb_tip[0] / 640  # Normalizează pe baza rezoluției camerei (ex. 640x480)
        thumb_y = thumb_tip[1] / 480

        # Afișează coordonatele pentru debugging
        print(f"Thumb coordinates (normalized): X={thumb_x:.2f}, Y={thumb_y:.2f}")

        # Mută cursorul folosind funcția de mai sus
        move_cursor(thumb_x, thumb_y)

    except IndexError:
        print("Landmark list does not contain enough points for thumb tip.")

def select_mode(key, mode):
    number = -1
    if ord('0') <= key <= ord('9'):
        number = key - ord('0')
    if key == 107:  # Tasta 'k'
        mode = 1
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_height, image_width = image.shape[:2]
    landmark_array = np.array([[int(landmark.x * image_width), int(landmark.y * image_height)] for landmark in landmarks.landmark])
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_height, image_width = image.shape[:2]
    landmark_point = [[min(int(landmark.x * image_width), image_width - 1), min(int(landmark.y * image_height), image_height - 1)] for landmark in landmarks.landmark]
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def logging_csv(number, mode, landmark_list):
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'x_y_values.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        for connection in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            image = cv.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]), (0, 255, 0), 2)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    return image

def draw_info(image, mode, number):
    cv.putText(image, "Mode:" + str(mode), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if 0 <= number <= 9:
        cv.putText(image, "Label:" + str(number), (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

def draw_info_text(image, brect, handedness, label):
    cv.rectangle(image, (brect[0], brect[1] - 30), (brect[2], brect[1]), (0, 255, 0), -1)
    cv.putText(image, handedness.classification[0].label + ':' + label, (brect[0] + 5, brect[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)
    return image

if __name__ == "__main__":
    main()
