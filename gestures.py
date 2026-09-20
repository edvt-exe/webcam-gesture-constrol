# Gesture classifiers and per-hand state tracking

import math
import collections

SWIPE_FRAMES = 10
FIST_TO_PALM_WINDOW = 0.6
VOL_MOVE_THRESH = 0.04
OK_DIST_THRESH = 0.05
SWIPE_THRESH = 0.35


class HandState:
    def __init__(self):
        self.wrist_x_hist = collections.deque(maxlen=SWIPE_FRAMES)
        self.index_y_hist = collections.deque(maxlen=5)
        self.index_pos_hist = collections.deque(maxlen=8)
        self.was_fist = False
        self.fist_time = 0.0
        self.last_action_time = 0.0


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_open_palm(up):
    return sum(up) >= 4


def is_fist(up):
    return sum(up) == 0


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


def circular_motion(pos_hist):
    # Detect circular index-finger motion and returns +1, -1 or 0
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