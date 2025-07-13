from ple.games.flappybird import FlappyBird
from ple import PLE
import time

game = FlappyBird()
env = PLE(game, fps=30, display_screen=True)
env.init()

for _ in range(70):
    reward  = env.act(env.getActionSet()[0])
    print(f"Reward: {reward}")
    frame = env.getScreenRGB()
    time.sleep(0.1)
    print(frame.shape)