# Webcam Hand Gesture PC Control

import cv2
import mediapipe as mp
import time
import collections
import pyautogui
import math

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

VOL_MOVE_THRESH = 0.04
OK_DIST_THRESH = 0.05

FIST_TO_PALM_WINDOW = 0.6
SCROLL_SENSITIVITY = 400

class HandState:
    def __init__(self):
        self.wrist_x_hist = collections.deque(maxlen=SWIPE_FRAMES)
        self.index_y_hist = collections.deque(maxlen=5)
        self.index_pos_hist = collections.deque(maxlen=8)
        self.was_fist = False
        self.fist_time = 0.0
        self.last_action_time = 0.0

def is_open_palm(up):
    return sum(up) >= 4

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def is_ok_sign(lm):
    return dist(lm[4], lm[8]) < OK_DIST_THRESH

def is_pointing_up(up):
    return up[1] and not any(up[2:])

def is_victory(up):
    return up[1] and up[2] and not up[3] and not up[4]

def palm_covers_face(lm):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    w_norm = max(xs) - min(xs)
    h_norm = max(ys) - min(ys)
    cx = (max(xs) + min(xs)) / 2
    return w_norm > 0.45 and h_norm > 0.45 and 0.25 < cx < 0.75

def is_fist(up):
    return sum(up) == 0

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

def circular_motion(pos_hist):
    # Detect circular index finger motion and returns +1, -1 or 0
    if len(pos_hist) < pos_hist.maxlen:
        return 0
    xs = [p[0] for p in pos_hist]
    ys = [p[1] for p in pos_hist]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    angles = [math.atan2(y - cy, x - cx) for x, y in pos_hist]

    total = 0
    for i in range(1, len(angles)):
        d = angles[i] - angles[i - 1]
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        total += d

    radius = math.hypot(xs[0] - cx, ys[0] - cy)
    if radius < 0.03:
        return 0
    if total > 2.5:
        return 1
    if total < -2.5:
        return -1
    return 0

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    states = {"Left": HandState(), "Right": HandState()}

    with mp_hands.Hands(model_complexity=0, max_num_hands=2,min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

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

                    index_tip = lm[8]
                    st.index_y_hist.append(index_tip.y)

                    st.index_pos_hist.append((index_tip.x, index_tip.y))

                    can_fire = (now - st.last_action_time) > COOLDOWN

                    # Fist -> open palm quick transition => Fullscreen toggle
                    if is_fist(up):
                        st.was_fist = True
                        st.fist_time = now
                    elif (gesture_label == "None" and is_open_palm(up) and st.was_fist
                          and (now - st.fist_time) < FIST_TO_PALM_WINDOW and can_fire):
                        pyautogui.press('f11')
                        gesture_label = "Fullscreen Toggle"
                        st.last_action_time = now
                        st.was_fist = False

                    # OK sign => Play/Pause
                    if gesture_label == "None" and is_ok_sign(lm) and can_fire:
                        pyautogui.press('space')
                        gesture_label = "Play/Pause"
                        st.last_action_time = now

                    # Palm covering face => Show Desktop
                    if gesture_label == "None" and palm_covers_face(lm) and can_fire:
                        pyautogui.hotkey('win', 'd')
                        gesture_label = "Show Desktop"
                        st.last_action_time = now

                    # Victory sign => Screenshot
                    if gesture_label == "None" and is_victory(up) and can_fire:
                        pyautogui.hotkey('win', 'shift', 's')
                        gesture_label = "Screenshot"
                        st.last_action_time = now

                    # Pointing finger vertical move => Volume up/down
                    if (gesture_label == "None" and is_pointing_up(up)
                            and len(st.index_y_hist) == st.index_y_hist.maxlen and can_fire):
                        dy = st.index_y_hist[0] - st.index_y_hist[-1]
                        if abs(dy) > VOL_MOVE_THRESH:
                            if dy > 0:
                                pyautogui.press('volumeup')
                                gesture_label = "Volume Up"
                            else:
                                pyautogui.press('volumedown')
                                gesture_label = "Volume Down"
                            st.last_action_time = now

                    # Circular index motion => Scroll
                    if gesture_label == "None" and is_pointing_up(up) and can_fire:
                        direction = circular_motion(st.index_pos_hist)
                        if direction != 0:
                            pyautogui.scroll(direction * SCROLL_SENSITIVITY)
                            gesture_label = "Scroll " + ("Up" if direction > 0 else "Down")
                            st.last_action_time = now
                            st.index_pos_hist.clear()

                    # Left hand open palm => Mute
                    if gesture_label == "None" and label == "Left" and is_open_palm(up) and can_fire:
                        pyautogui.press('volumemute')
                        gesture_label = "Mute Toggle"
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

                    # Raise 1-5 fingers => switch to tab N (fallback, lowest priority)
                    if gesture_label == "None" and 1 <= fcount <= 5 and can_fire:
                        pyautogui.hotkey('ctrl', str(fcount))
                        gesture_label = f"Switch to Tab {fcount}"
                        st.last_action_time = now

            draw_hud(frame, gesture_label, finger_count)
            cv2.imshow("Gesture PC Control", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()