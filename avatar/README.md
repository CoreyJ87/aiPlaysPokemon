# Companion Avatar for OBS

A floating one-eyed robot ball, rendered on an HTML canvas and controlled entirely from Python over a local WebSocket.

## How it works

```
your Python code  ──►  Avatar() WebSocket server (localhost:8765)  ──►  avatar.html in OBS
```

The Python side is the server and holds the avatar's current state. The browser source in OBS connects to it and auto-reconnects every second if the connection drops, so it doesn't matter whether you start OBS or your script first — whenever the page connects, it receives the full current state.

## Setup

1. `pip install websockets`
2. Run the demo to start the server: `python avatar_controller.py`
3. In OBS: **Sources → + → Browser**
   - Check **Local file** and select `avatar.html`
   - Width/Height: something like **400 × 400** (the avatar centers itself and scales its float within the canvas)
   - The page background is transparent, so it composites cleanly over your scene.
4. You should see the ball bobbing and cycling through moods from the demo loop.

## Using it from your own code

```python
from avatar_controller import Avatar

avatar = Avatar()                # starts the server in a background thread

avatar.set_mood("happy")         # neutral | happy | sad | afk | stressed
avatar.look(0.7, -0.2)           # x: -1 left..1 right, y: -1 up..1 down
avatar.set_eye_color("#4169e1")  # smooth color fade
avatar.set_float_speed(2.0)      # 0 = frozen, 1 = normal
avatar.stop_float()

# Or set everything at once:
avatar.set_state(mood="sad", eye_color="#4169e1", look=(0, 0.5), float_speed=0.5)

# Convenience presets (mood + matching color):
avatar.happy(); avatar.sad(); avatar.angry(); avatar.asleep(); avatar.neutral()
```

`Avatar()` is thread-safe and synchronous — call it from anywhere in your backend without touching asyncio.

## Pupil shapes by mood

| mood     | pupil            |
|----------|------------------|
| neutral  | filled circle    |
| happy    | upside-down U    |
| sad      | U                |
| afk      | horizontal line  |
| stressed | X                |

## Customizing the look

Open `avatar.html` and edit the `CONFIG` object at the top: circle radii, body color, float amplitude, how far the eye can wander, and the WebSocket URL/port (change the port in both files if 8765 is taken).

After editing, right-click the browser source in OBS → **Refresh cache of current page**.
