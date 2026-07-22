import threading # for multithreading
import time # for sleep
import sys # needed for sys path

import cv2 # needed for image processing
import numpy as np # needed for image processing

sys.path.append("tts") # add the tts folder to the system path so we can import the avatar_tts module
from avatar_tts import Avatar_TTS # for text to speech

sys.path.append("../mGBA") # add the mGBA folder to the system path so we can import the mgba_client module
from mgba_client import MGBAClient, MGBAError # for communicating with the mGBA emulator

from avatar_controller import Avatar # for controlling the avatar

class AvatarAI:
    """
    Class that handles the avatar AI
    """

    # tuning knobs for the gaze heuristic
    GAZE_WIDTH = 120        # width the screenshot is shrunk to before analysis (the GBA screen is 240x160)
    GAZE_BLUR = 25          # size of the box blur that turns single pixels into "areas of interest"
    MOTION_WEIGHT = 3.0     # how much more a moving pixel counts than a merely detailed one
    MOTION_FLOOR = 4.0      # ignore frame differences quieter than this, they're just dithering/palette noise
    GAZE_SMOOTHING = 0.6    # 0.0 snaps straight to the new target, 0.9 drifts very slowly

    def __init__(self):
        self.avatar = Avatar() # create an instance of the Avatar class
        self.avatar_tts = Avatar_TTS(voice="tts/en_US-amy-medium.onnx", output_file="tts/output.wav") # create an instance of the Avatar_TTS class
        self.mgba_client = MGBAClient() # create an instance of the MGBAClient class

        self._prev_frame = None # previous (downscaled, grayscale) screenshot, used to detect motion
        self._look_target = (0.0, 0.0) # last gaze direction, so the eyes ease towards the new one

        look_thread = threading.Thread(target=self.look_loop, daemon=True) # create a thread for the look loop
        look_thread.start() # start the look loop thread


    def say(self, text):
        """
        method to turn text into speech

        Args:
            text (string): The text to be converted to speech
        """
        self.avatar_tts.say(text) # call the say method of the Avatar_TTS class

    def look_loop(self):
        """
        Method for having the avatar call a screenshot, find the most intersting part of it, and look at that
        """

        while True:

            if self.avatar._state["mood"] == "afk": # if the avatar is afk, don't do anything
                time.sleep(1)
                continue

            else:
                # get the screenshot from the mgba emulator (returned as raw PNG bytes)
                screenshot = self.mgba_client.screenshot()

                # find the most interesting part of the screenshot
                interesting_part = self.find_most_interesting_part(screenshot)

                # look at the interesting part of the screenshot
                self.avatar.look(interesting_part[0], interesting_part[1])

                time.sleep(0.25) # sleep for a second before taking another screenshot

    def find_most_interesting_part(self, screenshot):
        """
        Method for finding the most interesting part of a screenshot

        Scores every area of the screen on how much detail it has (Canny edge density) and how
        much it just changed (difference against the previous frame), then picks the highest
        scoring spot. Motion is weighted heavily so the avatar follows the sprite that's moving
        rather than locking onto the text box, which is permanently the most edge-dense thing
        on screen.

        Args:
            screenshot (numpy array or bytes): The screenshot to be analyzed, either an image
                array or the raw PNG bytes returned by MGBAClient.screenshot()

        Returns:
            tuple: The (x, y) direction of the most interesting part, normalised for
                Avatar.look() — x is -1 (left) to 1 (right), y is -1 (up) to 1 (down)
        """
        frame = self._prepare_frame(screenshot)

        if frame is None: # unreadable screenshot, keep looking wherever we were looking
            return self._look_target

        # detail: edges, blurred out so a dense cluster of them scores higher than a lone line
        edges = cv2.Canny(frame, 80, 160)
        interest = self._normalise(cv2.blur(edges.astype(np.float32), (self.GAZE_BLUR, self.GAZE_BLUR)))

        # motion: what changed since last frame, blurred the same way
        if self._prev_frame is not None and self._prev_frame.shape == frame.shape:
            motion = cv2.absdiff(frame, self._prev_frame).astype(np.float32)
            motion = cv2.blur(motion, (self.GAZE_BLUR, self.GAZE_BLUR))

            if motion.max() > self.MOTION_FLOOR: # only trust it if something actually moved
                # both maps get scaled to 0..1 first, otherwise a big static block of detail
                # outweighs a small moving sprite just because it covers more pixels
                interest += self.MOTION_WEIGHT * self._normalise(motion)

        self._prev_frame = frame

        if interest.max() <= 0.0: # blank screen (a fade, a transition), hold the current gaze
            return self._look_target

        # the brightest spot in the score map is what we want to look at
        _, _, _, (peak_x, peak_y) = cv2.minMaxLoc(interest)

        height, width = interest.shape
        x = (peak_x / max(width - 1, 1)) * 2.0 - 1.0
        y = (peak_y / max(height - 1, 1)) * 2.0 - 1.0

        # ease towards the new target so the eyes glide instead of twitching between frames
        smoothing = self.GAZE_SMOOTHING
        self._look_target = (
            self._look_target[0] * smoothing + x * (1.0 - smoothing),
            self._look_target[1] * smoothing + y * (1.0 - smoothing),
        )

        return self._look_target

    def _normalise(self, score_map):
        """
        Scale a score map so its strongest spot is 1.0, leaving it flat if there's nothing in it

        Args:
            score_map (numpy array): The float map to be scaled

        Returns:
            numpy array: The same map scaled to the 0..1 range
        """
        peak = float(score_map.max())
        return score_map / peak if peak > 0.0 else score_map

    def _prepare_frame(self, screenshot):
        """
        Turn whatever the emulator handed us into a small grayscale image ready for analysis

        Args:
            screenshot (numpy array or bytes): The screenshot to be converted

        Returns:
            numpy array: A downscaled single channel image, or None if it couldn't be decoded
        """
        if isinstance(screenshot, (bytes, bytearray)): # raw PNG bytes off the socket
            screenshot = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_GRAYSCALE)

        if screenshot is None or getattr(screenshot, "size", 0) == 0:
            return None

        if screenshot.ndim == 3: # colour image, drop it to grayscale
            channels = screenshot.shape[2]
            if channels == 4:
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
            elif channels == 3:
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            else:
                screenshot = screenshot[:, :, 0]

        # shrink it down, both to make the blurs cheap and to stop single pixels dominating
        height, width = screenshot.shape[:2]
        scale = self.GAZE_WIDTH / float(width)

        if scale < 1.0:
            screenshot = cv2.resize(
                screenshot,
                (self.GAZE_WIDTH, max(int(round(height * scale)), 1)),
                interpolation=cv2.INTER_AREA,
            )

        return screenshot





if __name__ == "__main__":
    print("Avatar AI Test")

    avatar_ai = AvatarAI()

    while True:
        text = input("Enter text to convert to speech (or 'exit' to quit): ")
        if text.lower() == "exit":
            print("Exiting...")
            break
        avatar_ai.say(text)