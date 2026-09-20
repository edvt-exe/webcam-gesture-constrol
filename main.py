# Webcam Hand Gesture PC Control

import time
import cv2
import mediapipe as mp
import pyautogui

from hand_tracker import fingers_up
from gestures import ( HandState, is_open_palm, is_fist, is_ok_sign, is_pointing_up, is_victory, palm_covers_face, circular_motion, SWIPE_THRESH, FIST_TO_PALM_WINDOW, VOL_MOVE_THRESH)
from hud import draw_hud

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

CAM_INDEX = 0
COOLDOWN = 1.2
SCROLL_SENSITIVITY = 400


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    states = {"Left": HandState(), "Right": HandState()}
    global_last_action = 0.0
    prev_time = time.time()

    with mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
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

                    # Fist => open palm quick transition => Fullscreen toggle
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

                    if gesture_label != "None":
                        global_last_action = now

            cooldown_remaining = max(0.0, COOLDOWN - (now - global_last_action))
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now

            draw_hud(frame, gesture_label, finger_count, cooldown_remaining, fps)
            cv2.imshow("Gesture PC Control", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()