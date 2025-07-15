from stable_baselines3.common.callbacks import BaseCallback

class EntropyAnnealCallback(BaseCallback):
    def __init__(self, initial_p: float, final_p: float, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.initial_p = initial_p
        self.final_p = final_p
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Linear anneal
        progress_remaining = 1.0 - (self.num_timesteps / self.total_timesteps)
        current_ent_coef = self.final_p + (self.initial_p - self.final_p) * progress_remaining
        self.model.ent_coef = current_ent_coef
        return True
