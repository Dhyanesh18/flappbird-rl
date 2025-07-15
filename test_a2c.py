from stable_baselines3 import A2C
from flappy_gym_env import FlappyBirdGymEnv
import time

# Create environment
env = FlappyBirdGymEnv(display_screen=True)

# Load the trained A2C model
model = A2C.load("a2c_flappybird")  # Saved A2C model filename

# Reset the environment
obs, _ = env.reset()
done = False

# Run the agent until it crashes
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    env.render()
    time.sleep(0.03)

# Close the environment window after the episode ends
env.close()
