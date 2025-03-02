# Hand Gesture Control Application

## Overview
This application allows users to control their PC using hand gestures. It leverages MediaPipe for real-time hand tracking and maps specific gestures to various mouse actions such as hover, right-click, and left-click.

## Features
- **Hand Tracking**: Uses MediaPipe's hand detection model to recognize hand gestures.
- **Mouse Control**: Perform PC interactions using gestures:
  - **Hover**: Move the cursor by moving your hand.
  - **Left Click**: Perform a left click using a specific gesture.
  - **Right Click**: Perform a right click using another predefined gesture.
- **Real-time Processing**: Fast and efficient hand tracking using OpenCV and MediaPipe.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- PyAutoGUI (`pyautogui`)
- Numpy (`numpy`)

## Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/your-repo/hand-gesture-control.git](https://github.com/sHaises/EasyControl)
  
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe pyautogui numpy
   ```

## Usage
Run the application with:
```bash
python main.py
```

### Gesture Mappings
| Gesture         | Action         |
|----------------|---------------|
| Open Palm      | Cursor Move    |
| Closed palm | Left Click     |
| Index finger up| Right Click  |


## Contributions
Feel free to open an issue or pull request to contribute!

