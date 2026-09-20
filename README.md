# Gesture PC Control

Control your PC with hand gestures picked up by your webcam. No extra hardware, just OpenCV + MediaPipe doing the hand tracking and PyAutoGUI firing off the actual keypresses.

I built this mostly because I got tired of reaching for the keyboard every time I wanted to skip a song or take a screenshot while coding. Turned into a decent excuse to mess around with MediaPipe's hand landmarks properly.

## What it does

Point your hand at the camera and it reads finger positions in real time to trigger stuff on your PC:

- Swipe your palm left/right → Alt+Tab
- Hold up 1 to 5 fingers → jump straight to browser tab 1-5
- Close your fist then open it fast → toggle fullscreen (F11)
- Point one finger and move it up/down → volume up/down
- Touch your thumb and index tip together (OK sign) → play/pause
- Raise your left palm → mute
- Peace sign → screenshot (Win+Shift+S)
- Cover your face/camera with your palm → show desktop
- Draw a little circle in the air with your index finger → scroll up/down

Everything has a cooldown so you don't accidentally trigger five actions in one second while your hand is mid-gesture.

## Setup

You'll need Python 3.9+ and a webcam.

```bash
pip install -r requirements.txt
python main.py
```

A window pops up showing your camera feed with the hand landmarks drawn on top, plus a small HUD in the corner showing the detected gesture, finger count, and cooldown. Press `Q` or `ESC` to close it.

## How the detection works

No ML model, no training — everything's derived straight from MediaPipe's 21 hand landmarks using basic geometry (distances, angles, y-position comparisons between finger tips and their joints). It's the "dumb but fast" approach and it's good enough for this kind of thing.

A couple of things that took some tweaking to get right:

- **Fist detection** counts how many fingers are curled, not raised — simple but works fine as long as the hand is fully in frame.
- **Circular scroll** tracks the last 8 positions of your index fingertip and sums up the angle change between consecutive points to figure out if you're drawing a circle and in which direction. Radius has a minimum threshold too, otherwise tiny hand jitter reads as a circle.
- Gesture checks have a fixed priority order (fist→palm first, tab-switch last as a fallback) because a few gestures share the same finger count and would otherwise fight each other for the same frame.

## Known limitations

- Lighting matters a lot — in a dark room the hand tracking gets flaky.
- `Ctrl+1..5` for tab switching only works in browsers that actually bind those shortcuts (Chrome, Firefox). Doesn't do anything meaningful elsewhere.
- Face-covering detection for "show desktop" is just a rough bounding-box check, so getting your hand too close to the camera for any reason can trigger it by accident.
- One camera, so if two hands are doing conflicting gestures at once, whichever gets evaluated first wins.

## Project structure
main.py # camera loop, MediaPipe setup, gesture priority logic

Might split this into separate modules later if it grows (hand tracking / gesture classifiers / HUD are decent seams to cut along), but for a project this size one file was easier to iterate on.

## Things I'd add next

- Config file (YAML or JSON) for tweaking thresholds instead of hardcoding constants at the top — swipe distance, cooldown, scroll sensitivity are all things that'll vary a lot depending on camera and distance from it.
- Per-user calibration step on startup (hold your hand still for 3 seconds, adjust thresholds based on your actual hand size/distance).
- Multi-monitor awareness — right now everything just assumes a single active window/display.
- Swap the face-covering check for something less prone to false positives, maybe actual face landmark detection instead of a bounding-box heuristic.
- Customizable gesture-to-action bindings instead of everything hardcoded in `main.py`.
- Small on-screen gesture guide/cheatsheet toggle, so I don't have to keep checking this README to remember what does what.