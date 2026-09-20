# Webcam Hand Gesture PC Control

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

CAM_INDEX = 0

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    with mp_hands.Hands(model_complexity=0, max_num_hands=2,min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, freame = cap.read()
            if not ok:
                break

            frame = cv2.flip(freame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Gesture PC Control", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()