import csv
import copy
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui
import time
from handgest import HandGestureRecognition


pyautogui.FAILSAFE = False

#we save the last positions of the cursoure in order to have a more fluid movment
previous_x, previous_y = None, None
cursor_smooth_factor = 0.2  # 0.2 will smooth out the cursor movements

# Variables for click detection
last_click_time = 0  # to manage debounce for clicks (in order not to have multiple unwanted clicks)
left_click_hold_start_time = None  # to track how long left-click gesture is held (this variable is going to be used in order to determin when to make a double click)

def main():
    # Declare global variables for tracking cursor and click states
    global previous_x, previous_y, last_click_time, left_click_hold_start_time

    # Video capture device and configuration settings
    cap_device = 0  # Default camera device index
    cap_width = 960  # Desired width of the video frame
    cap_height = 540  # Desired height of the video frame

    # Mediapipe Hand Detection configuration
    use_static_image_mode = False  # Whether to use static images or continuous video
    min_detection_confidence = 0.7  # Minimum confidence for hand detection
    min_tracking_confidence = 0.5  # Minimum confidence for tracking landmarks
    use_brect = True  # Whether to use bounding rectangles for visualization

    # Initialize video capture with the specified device
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Initialize Mediapipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,  # Maximum number of hands to detect
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Initialize hand gesture recognition
    hand_gesture_recognition = HandGestureRecognition()

    # Read labels for gesture recognition from a CSV file
    with open('labels.csv', encoding='utf-8-sig') as f:
        labels = csv.reader(f)
        labels = [row[0] for row in labels]  # Extract labels from the CSV file

    mode = 0  # Initialize mode variable

    while True:
        # Wait for key press (10ms delay) and check if ESC is pressed to exit
        key = cv.waitKey(10)
        if key == 27:  # ESC key
            break

        # Update mode and number based on key press
        number, mode = select_mode(key, mode)

        # Capture a frame from the camera
        ret, image = cap.read()
        if not ret:
            break  # Exit loop if no frame is captured

        # Flip the image horizontally for a mirror effect
        image = cv.flip(image, 1)

        # Create a copy of the image for debug purposes
        debug_image = copy.deepcopy(image)

        # Convert the image to RGB format for Mediapipe processing
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False  # Optimize for Mediapipe processing
        results = hands.process(image)  # Detect hands and landmarks
        image.flags.writeable = True  # Re-enable writing

        if results.multi_hand_landmarks is not None:
            # Process each detected hand
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Calculate bounding rectangle for the hand
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Convert hand landmarks to pixel coordinates
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # Preprocess the landmarks for recognition
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                # Log the data if required
                logging_csv(number, mode, pre_processed_landmark_list)

                # Recognize the hand gesture
                hand_sign_id = hand_gesture_recognition(pre_processed_landmark_list)

                # Handle specific gestures
                if hand_sign_id == 0:  # Gesture for moving the cursor
                    move_cursor_from_landmarks(landmark_list)

                if hand_sign_id == 1:  # Gesture for left-click
                    if left_click_hold_start_time is None:
                        left_click_hold_start_time = time.time()  # Start hold timer

                    hold_duration = time.time() - left_click_hold_start_time
                    if hold_duration > 5:  # If hold duration exceeds 5 seconds
                        pyautogui.doubleClick(button='left')  # Perform a double-click
                        left_click_hold_start_time = None  # Reset hold timer
                    elif time.time() - last_click_time > 0.5:  # Single-click with debounce
                        pyautogui.click(button='left')
                        last_click_time = time.time()
                else:
                    left_click_hold_start_time = None  # Reset hold timer if gesture is interrupted

                if hand_sign_id == 2:  # Gesture for right-click
                    if time.time() - last_click_time > 0.5:  # Debounce for right-click
                        pyautogui.click(button='right')
                        last_click_time = time.time()

                # Draw information on the debug image
                if 0 <= hand_sign_id < len(labels):  # If hand_sign_id is valid
                    debug_image = draw_info_text(debug_image, brect, handedness, labels[hand_sign_id])
                else:
                    debug_image = draw_info_text(debug_image, brect, handedness, "Unknown")

                # Draw the bounding rectangle and landmarks
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

        # Draw mode and label information on the debug image
        debug_image = draw_info(debug_image, mode, number)

        # Display the debug image in a window
        cv.imshow('Hand Gesture Recognition', debug_image)

    # Release resources and close the video window
    cap.release()
    cv.destroyAllWindows()



def move_cursor(x, y):
    # Declare the variables previous_x and previous_y as global
    # These variables will store the cursor's previous position to allow smooth movement
    global previous_x, previous_y
    
    # Get the dimensions of the screen (width and height in pixels)
    screen_width, screen_height = pyautogui.size()  # Screen dimensions
    
    # Ensure that the input coordinates (x, y) are in the range [0, 1]
    # If x or y is less than 0, it will be set to 0; if greater than 1, it will be set to 1
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    # Map the relative coordinates (x, y) to absolute screen coordinates
    # Multiply by screen dimensions to get pixel values
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)

    # Stabilize the cursor movement for smooth transitions
    if previous_x is None or previous_y is None:
        # If this is the first call, initialize the previous positions to the current positions
        previous_x, previous_y = screen_x, screen_y

    # Calculate the smoothed cursor positions by applying a smoothing factor
    # This creates a gradual transition from the previous position to the new position
    screen_x = int(previous_x + cursor_smooth_factor * (screen_x - previous_x))
    screen_y = int(previous_y + cursor_smooth_factor * (screen_y - previous_y))

    # Ensure that the smoothed cursor positions do not exceed the screen boundaries
    # Clip the x and y coordinates to stay within [0, screen_width-1] and [0, screen_height-1]
    screen_x = np.clip(screen_x, 0, screen_width - 1)
    screen_y = np.clip(screen_y, 0, screen_height - 1)

    # Move the cursor to the calculated screen coordinates
    pyautogui.moveTo(screen_x, screen_y)

    # Update the previous cursor position with the current position
    # This ensures continuity in smoothing calculations for the next movement
    previous_x, previous_y = screen_x, screen_y


def move_cursor_from_landmarks(landmark_list):
    
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

# Function to select mode and number based on key input
def select_mode(key, mode):
    number = -1
    # Check if key corresponds to a numeric character (0-9)
    if ord('0') <= key <= ord('9'):
        number = key - ord('0')  # Convert ASCII value to integer
    if key == 107:  # Check if key 'k' is pressed
        mode = 1
    return number, mode

# Function to calculate bounding rectangle around landmarks
def calc_bounding_rect(image, landmarks):
    image_height, image_width = image.shape[:2]
    # Convert landmark coordinates to pixel values
    landmark_array = np.array([[int(landmark.x * image_width), int(landmark.y * image_height)] for landmark in landmarks.landmark])
    # Compute the bounding rectangle
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

# Function to calculate a list of landmarks in pixel coordinates
def calc_landmark_list(image, landmarks):
    image_height, image_width = image.shape[:2]
    # Ensure coordinates are within image bounds
    landmark_point = [[min(int(landmark.x * image_width), image_width - 1), min(int(landmark.y * image_height), image_height - 1)] for landmark in landmarks.landmark]
    return landmark_point

# Function to preprocess landmarks for normalization
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)  # Create a copy to avoid modifying the original
    base_x, base_y = 0, 0
    # Normalize landmark positions relative to the first point
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    # Flatten the list of landmarks
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))  # Find the maximum absolute value for normalization

    # Normalize all values
    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# Function to log landmarks and their associated number to a CSV file
def logging_csv(number, mode, landmark_list):
    if mode == 1 and (0 <= number <= 9):  # Only log when mode is 1 and number is valid
        csv_path = 'data.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])  # Write the number and flattened landmark list

# Function to draw landmarks on the image
def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Draw lines connecting specific landmark points
        for connection in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            image = cv.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]), (0, 255, 0), 2)
    return image

# Function to draw a bounding rectangle on the image
def draw_bounding_rect(use_brect, image, brect):
    if use_brect:  # Check if bounding rectangle should be drawn
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    return image

# Function to display mode and label information on the image
def draw_info(image, mode, number):
    cv.putText(image, "Mode:" + str(mode), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if 0 <= number <= 9:  # Display label if the number is valid
        cv.putText(image, "Label:" + str(number), (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

# Function to display additional information about handedness and label
def draw_info_text(image, brect, handedness, label):
    # Draw a filled rectangle to serve as the background for the text
    cv.rectangle(image, (brect[0], brect[1] - 30), (brect[2], brect[1]), (0, 255, 0), -1)
    # Display handedness and label text
    cv.putText(image, handedness.classification[0].label + ':' + label, (brect[0] + 5, brect[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)
    return image


if __name__ == "__main__":
    main()
