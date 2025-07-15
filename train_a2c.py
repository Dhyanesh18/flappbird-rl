from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from flappy_gym_env import FlappyBirdGymEnv
from stable_baselines3.common.monitor import Monitor

# Create the environment
env = FlappyBirdGymEnv(frame_skip=2, frame_stack=4, width=84, height=84)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Check the env
check_env(env.envs[0], warn=True)

# A2C Model
model = A2C(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./a2c_flappybird_tensorboard/",
    n_steps=32,
    learning_rate=7e-4,
    gamma=0.99,
    ent_coef=0.02,
    vf_coeff=0.5,
    max_grad_norm=0.5,
    device="cuda"
)

# Train the agent
model.learn(total_timesteps=5_000_000)

# Save the agent 
model.save("a2c_flappybird")

print("Training complete. Model saved as 'a2c_flappybird'.")