# OpenCV HUD overlay rendering

import cv2

def draw_hud(frame, gesture, finger_count):
    cv2.rectangle(frame, (0, 0), (260, 70), (20, 20, 20), -1)
    cv2.putText(frame, f"Gesture: {gesture}", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
    cv2.putText(frame, f"Fingers: {finger_count}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)