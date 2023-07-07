import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import subprocess
import screeninfo

# Get screen resolution
screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

# Create a named window for the camera feed
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

# Set the window to fullscreen
cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Function to toggle window state between fullscreen and normal
def toggle_window_state():
    current_state = cv2.getWindowProperty('Image', cv2.WND_PROP_FULLSCREEN)
    if current_state == cv2.WINDOW_FULLSCREEN:
        cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    else:
        cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.CAP_GSTREAMER = True
cv2.CAP_FFMPEG = True
cv2.CAP_DSHOW = True
cap = cv2.VideoCapture(2)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def set_volume(vol):
    # to adjust the system volume
    subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{vol}%"])
    
    # to show volume bar in camera window
    # subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{vol}%"])

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList != []:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger

        cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)
        vol = int(np.interp(length, [30, 350], [0, 100]))  # Map length to volume range

        set_volume(vol)

        # Draw volume bar
        # cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)  # Volume bar outline
        # cv2.rectangle(img, (50, int(400 - vol * 2.5)), (85, 400), (0, 0, 255), cv2.FILLED)  # Filled portion of volume bar
        # cv2.putText(img, f"{vol}%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 98), 3)  # Volume percentage text

        cv2.imshow('Image', img)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # Press 'm' to toggle between fullscreen and normal window state
    if key == ord('m'):
        toggle_window_state()

    # Press 'q' to exit the program
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
