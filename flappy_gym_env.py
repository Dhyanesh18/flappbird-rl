import gymnasium as gym
import numpy as np
import cv2
from ple.games.flappybird import FlappyBird
from ple import PLE
from gymnasium import spaces
from collections import deque


class FlappyBirdGymEnv(gym.Env):
    def __init__(self, frame_skip=4, frame_stack=4, width=84, height=84, display_screen=True):
        super().__init__()
        self.game = FlappyBird()
        self.env = PLE(self.game, fps=30, display_screen=display_screen)
        self.env.init()

        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.width = width
        self.height = height

        # Actions: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)

        # Observation space: 4 stacked uint8 grayscale frames in (C, H, W)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.frame_stack, self.height, self.width),
            dtype=np.uint8
        )

        self.frames = deque(maxlen=self.frame_stack)

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)  # No normalization

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.env.reset_game()
        self.frames.clear()

        frame = self.env.getScreenRGB()
        processed = self.preprocess(frame)

        for _ in range(self.frame_stack):
            self.frames.append(processed)

        obs = np.array(self.frames, dtype=np.uint8)
        return obs, {}

    def step(self, action):
        reward = 0.0
        done = False

        ple_action = self.env.getActionSet()[action]

        for _ in range(self.frame_skip):
            reward += self.env.act(ple_action)
            done = self.env.game_over()
            if done:
                break

        frame = self.env.getScreenRGB()
        processed = self.preprocess(frame)
        self.frames.append(processed)

        # living bonus:
        reward += 0.1

        # larger penalty for crashing
        if done:
            reward -= 5

        obs = np.array(self.frames, dtype=np.uint8)
        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, {}


    def render(self):
        self.env.display_screen = True

    def close(self):
        self.env.display_screen = False
