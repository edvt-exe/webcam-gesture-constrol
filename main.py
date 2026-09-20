# Webcam Hand Gesture PC Control

import cv2
import mediapipe as mp
import time
import collections
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

CAM_INDEX = 0
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

COOLDOWN = 1.2
SWIPE_THRESH = 0.35
SWIPE_FRAMES = 10

class HandState:
    def __init__(self):
        self.wrist_x_hist = collections.deque(maxlen=SWIPE_FRAMES)
        self.last_action_time = 0.0

def is_open_palm(up):
    return sum(up) >= 4

def draw_hud(frame, gesture, finger_count):
    cv2.rectangle(frame, (0, 0), (260, 70), (20, 20, 20), -1)
    cv2.putText(frame, f"Gesture: {gesture}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
    cv2.putText(frame, f"Fingers: {finger_count}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    states = {"Left": HandState(), "Right": HandState()}

    with mp_hands.Hands(model_complexity=0, max_num_hands=2,min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, freame = cap.read()
            if not ok:
                break

            frame = cv2.flip(freame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            finger_count = 0
            gesture_label = "None"
            now = time.time()

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = hd.classification[0].label
                    lm = hand_lms.landmark
                    st = states[label]

                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                    up = fingers_up(lm, label)
                    fcount = sum(up)
                    finger_count = fcount

                    wrist = lm[0]
                    st.wrist_x_hist.append(wrist.x)

                    can_fire = (now - st.last_action_time) > COOLDOWN

                    # Raise 1-5 fingers => switch to tab N
                    if 1 <= fcount <= 5 and can_fire:
                        pyautogui.hotkey('ctrl', str(fcount))
                        gesture_label = f"Switch to Tab {fcount}"
                        st.last_action_time = now

                    # Swipe left/right with open palm => Alt+Tab
                    if (gesture_label == "None" and is_open_palm(up)
                            and len(st.wrist_x_hist) == st.wrist_x_hist.maxlen and can_fire):
                        dx = st.wrist_x_hist[-1] - st.wrist_x_hist[0]
                        if abs(dx) > SWIPE_THRESH:
                            pyautogui.hotkey('alt', 'tab')
                            gesture_label = "Swipe " + ("Right" if dx > 0 else "Left")
                            st.last_action_time = now
                            st.wrist_x_hist.clear()

            draw_hud(frame, gesture_label, finger_count)
            cv2.imshow("Gesture PC Control", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()