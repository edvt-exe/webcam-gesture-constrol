# Finger state and landmark helpers built on MediaPipe hand landmarks

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


def fingers_up(lm, handedness):
    # return list[bool] for [thumb, index, middle, ring, pinky]
    up = []
    if handedness == "Right":
        up.append(lm[FINGER_TIPS[0]].x < lm[FINGER_PIPS[0]].x)
    else:
        up.append(lm[FINGER_TIPS[0]].x > lm[FINGER_PIPS[0]].x)

    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        up.append(lm[tip].y < lm[pip].y)
    return up