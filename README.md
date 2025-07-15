# FlappyBirdRL 🐦

A Reinforcement Learning (RL) agent that learns to play Flappy Bird using **Stable Baselines3** and a custom **OpenAI Gym** environment.

This project demonstrates **deep RL** for an arcade-style game, with PPO/A2C and entropy annealing for improved exploration.

## Test clip
<img width="300" height="550" alt="image" src="https://github.com/user-attachments/assets/e04a5912-2c25-4208-9413-15cb2a16870b" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img width="300" height="550" alt="image" src="https://github.com/user-attachments/assets/ecbbc0bb-6fc8-4d72-b5a2-f7682990d42c" />


## Training graph (PPO)
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/c4bea6b6-2c95-46a0-b44e-0e85ec249aa4" />


---
## My setup and work

- The agents were trained on visual mode of flappy bird rather than numerical values of velocity, position etc.., so that its same as how humans percive the game.  
- 4 frames are stacked while passing through the CNN so that the agent understands the temporal information from the environment.  
- Entropy coefficient annealing is done so that the model stops exploring and starts exploiting at the later half of the training.  
- The config in the code files are what were used to get the best results.  
- Sadly, hyperparameter tuning wasn't possible and mostly intuition based tuning was done as the training of an agent for 10M timesteps took 19.2 Hrs.

## 📌 Project Highlights

- 🧠 **Algorithms:** PPO & A2C from Stable Baselines3
- 🎮 **Custom Gym Env:** Pixel-based Flappy Bird with frame skipping & stacking
- 🔬 **Entropy Annealing:** Controls exploration dynamically
- 📊 **TensorBoard:** Visualize training progress
- 🚀 **GPU Acceleration:** CUDA enabled

---

## 📂 Directory Structure

FlappyBirdRL/  
│  
├── flappy_gym_env.py # Custom Gym env  
├── train_ppo.py # PPO training script  
├── train_a2c.py # A2C training script  
├── entropy_annealing.py # Custom callback for entropy scheduling  
├── ppo_flappybird_tensorboard/ # Logs  
├── saved_models/ # Saved weights  
├── README.md # This file!  



---

## ⚙️ Installation

1. **Clone the repo**
   ```
   git clone https://github.com/your-username/FlappyBirdRL.git
   cd FlappyBirdRL

2. **Create a virtual environment (recommended)**
  ```
  python -m venv venv
  source venv/bin/activate  # Linux/macOS
  venv\Scripts\activate     # Windows
  ```
 
3. **Install dependencies**
  ```
  pip install -r requirements.txt
  ```

Note: The pygame-learning-environment package is to be installed the following way:
  ```
  git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
  cd PyGame-Learning-Environment
  pip install -e .
  ```


## Train the Agent
  ```
  python train_ppo.py
  ```
Edit train_ppo.py or train_a2c.py to tweak hyperparameters:

    n_steps, batch_size, gamma, learning_rate

    Entropy annealing: initial vs. final ent_coef

    Total timesteps

## 📈 Monitor Training
  ```
  tensorboard --logdir ppo_flappybird_tensorboard/
  ```

Open http://localhost:6006 in your browser to view learning curves, rewards, entropy, loss terms, etc.

## Test the Agent
  ```
  python test_ppo.py
  ```

## Acknowledgements

    Stable Baselines3

    OpenAI Gym

    Original Flappy Bird graphics by dotGBA
