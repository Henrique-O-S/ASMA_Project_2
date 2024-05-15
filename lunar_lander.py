import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.evaluation import evaluate_policy

model_dir = "/models/v01"

#env = gym.make("LunarLander-v2", continuous=False)

env = make_vec_env("LunarLander-v2", n_envs=16, monitor_dir=model_dir)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="/tb_logs/")

# Load the model
# loaded_model = PPO.load("lunar_lander_model")

def agent_eval(agent, n_eval_episodes=25, seed=42, verbose=1):
    eval_env = make_vec_env("LunarLander-v2", n_envs=16, seed=seed)
    reward_length = evaluate_policy(agent, eval_env, 
                           return_episode_rewards=True,
                           n_eval_episodes=n_eval_episodes,
                           deterministic=True)
    for i, p in enumerate(["mean", "length"]):
        mean = np.mean(reward_length[i])
        if verbose:
            sem = st.sem(reward_length[i])
            mean_CI = st.t.interval(0.95, df=len(reward_length[i])-1, loc=mean, scale=sem) 
        
            print(f"mean_{p}={mean:.3f}, SEM={sem}, CI=[{mean_CI[0]:.3f}:{mean_CI[1]:.3f}]")

    return np.mean(reward_length[0])

# Define the callback function
class CustomCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.done_eps = 0

    def _check(self):
        results = load_results(self.log_dir)
        new_eps = len(results) - self.done_eps
        self.done_eps = len(results)
        x, y = ts2xy(results, 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-new_eps:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f}, eval_episodes: {new_eps}, total_steps: {self.num_timesteps}, total_episodes: {self.done_eps}")
                    self.model.save(self.save_path)

    def _init_callback(self) -> None:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # if self.n_calls % self.check_freq == 0:
        #   self._check()
        return True

    def _on_training_end(self) -> None:
        interim_mean = agent_eval(self.model.load(self.save_path), verbose=0)
        final_mean = agent_eval(self.model, verbose=0)

        if interim_mean > final_mean:
            print(f"Interim model was better.")
            self.model.set_parameters(self.save_path)

        self.model.save(self.log_dir)

    def _on_rollout_end(self) -> None:
        self._check()

# Create the callback object
callback_object = CustomCallback(log_dir=model_dir)

# Train the model
model.learn(total_timesteps=100000, callback=callback_object, tb_log_name="v01_100k")

vec_env = model.get_env()

# Initialize total rewards
total_rewards = 0

# Run the environment with the loaded model
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    total_rewards += rewards
    #print(f"Step: {i}, Reward: {rewards}")
    if dones.any():
        obs = vec_env.reset()
        print(f"Episode finished. Total rewards: {total_rewards}")
        total_rewards = 0