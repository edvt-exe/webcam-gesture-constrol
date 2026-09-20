# Webcam Hand Gesture PC Control

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

CAM_INDEX = 0
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

def fingers_up(lm, handedness):
    # return a list[bool] for [thumb, index, middle, ring, pinky]:
    up = []
    if handedness == "Right":
        up.append(lm[FINGER_TIPS[0]].x < lm[FINGER_PIPS[0]].x)
    else:
        up.append(lm[FINGER_TIPS[0]].x > lm[FINGER_PIPS[0]].x)

    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        up.append(lm[tip].y < lm[pip].y)  
    return up

def draw_finger_count(frame, count):
    cv2.rectangle(frame, (0, 0), (200, 50), (20, 20, 20), -1)
    cv2.putText(frame, f"Fingers: {count}", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)

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

            finger_count = 0
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = hd.classification[0].label
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    up = fingers_up(hand_lms.landmark, label)
                    finger_count = sum(up)

            draw_finger_count(frame, finger_count)
            cv2.imshow("Gesture PC Control", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()