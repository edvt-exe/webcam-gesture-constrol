# OpenCV HUD overlay rendering

import cv2

def draw_hud(frame, gesture, finger_count, cooldown_remaining, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Gesture: {gesture}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
    cv2.putText(frame, f"Fingers: {finger_count}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cd_color = (0, 0, 255) if cooldown_remaining > 0 else (0, 255, 0)
    cd_text = f"Cooldown: {cooldown_remaining:.1f}s" if cooldown_remaining > 0 else "Ready"
    cv2.putText(frame, cd_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cd_color, 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)