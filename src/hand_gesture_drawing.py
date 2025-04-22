import cv2
import mediapipe as mp
import numpy as np
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open Camera
cap = cv2.VideoCapture(0)

# Create a blank canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = None, None
drawing = False  # Flag to track if drawing is active

def is_one_finger_up(hand_landmarks):
    """Check if only the index finger is up while others are down."""
    index_tip = hand_landmarks.landmark[8].y
    middle_tip = hand_landmarks.landmark[12].y
    ring_tip = hand_landmarks.landmark[16].y
    pinky_tip = hand_landmarks.landmark[20].y
    return index_tip < middle_tip and index_tip < ring_tip and index_tip < pinky_tip

def is_palm_open(hand_landmarks):
    """Detect if the palm is open by checking if all fingers are extended."""
    index_tip = hand_landmarks.landmark[8].y
    middle_tip = hand_landmarks.landmark[12].y
    ring_tip = hand_landmarks.landmark[16].y
    pinky_tip = hand_landmarks.landmark[20].y
    wrist = hand_landmarks.landmark[0].y
    return (index_tip < wrist and middle_tip < wrist and ring_tip < wrist and pinky_tip < wrist)

def get_erase_radius(hand_landmarks):
    """Estimate erasing radius based on palm size."""
    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]

    # Euclidean distance to estimate palm size
    distance = np.linalg.norm(
        np.array([wrist.x, wrist.y]) - np.array([middle_tip.x, middle_tip.y])
    )

    return int(distance * 300)  # Scale factor for real-world size

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Flip for better alignment
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger = hand_landmarks.landmark[8]
            palm_center = hand_landmarks.landmark[9]  # Palm base as center
            x, y = int(index_finger.x * 640), int(index_finger.y * 480)
            px, py = int(palm_center.x * 640), int(palm_center.y * 480)

            if is_palm_open(hand_landmarks):
                # Erase only the area under the palm
                erase_radius = get_erase_radius(hand_landmarks)
                cv2.circle(canvas, (px, py), erase_radius, (0, 0, 0), -1)
                logging.info(f"Erasing at {px}, {py} with radius {erase_radius}")

            elif is_one_finger_up(hand_landmarks):
                if prev_x is not None and prev_y is not None and drawing:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 2)
                drawing = True
                prev_x, prev_y = x, y

            else:
                drawing = False
                prev_x, prev_y = None, None

    # Show the camera feed
    cv2.imshow('Camera Feed', frame)
    # Show the drawing canvas
    cv2.imshow('Drawing Canvas', canvas)

    # Press 'c' to clear the canvas manually
    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        logging.info("Canvas cleared.")

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







