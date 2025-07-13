from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from flappy_gym_env import FlappyBirdGymEnv
from stable_baselines3.common.monitor import Monitor

env = FlappyBirdGymEnv(frame_skip=4, frame_stack=4, width=84, height=84)
env = Monitor(env)
env = DummyVecEnv([lambda: env]) 

check_env(env.envs[0], warn=True)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_flappybird_tensorboard/",
    n_steps=4096,
    batch_size=64,
    n_epochs=10,
    learning_rate=1e-4,
    gamma=0.99,
    ent_coef = 0.01,
    device="cuda" 
)

model.learn(total_timesteps=200_000)

model.save("ppo_flappybird")

print("Training complete. Model saved as 'ppo_flappybird'.")