import streamlit as st
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import json
from datetime import datetime
import hashlib
# 确保只导入必需的库

# ---------------------------------
# Streamlit/Colab 兼容的打印函数
# ---------------------------------
def print_status(message):
    """Prints status messages, checking if running in Streamlit."""
    if 'streamlit' in st.__doc__:
        st.info(message)
    else:
        print(message)

# ---------------------------------
# 训练日志回调函数
# ---------------------------------
class LoggingCallback(BaseCallback):
    """
    Saves training metrics to a list for POUW inclusion.
    """
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.logs = []

    def _on_step(self) -> bool:
        # 每 1000 步记录一次日志
        if self.n_calls % 1000 == 0:
            avg_reward = np.mean(self.locals['rewards']) if len(self.locals['rewards']) > 0 else 0
            self.logs.append({
                'timesteps': self.num_timesteps,
                'avg_reward': float(f'{avg_reward:.2f}')
            })
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}/{self.locals['total_timesteps']} | Avg Reward: {avg_reward:.2f}")
        return True

# ---------------------------------
# 2. RL 环境定义 (SmartLogisticsNavEnv)
# ---------------------------------
class SmartLogisticsNavEnv(gym.Env):
    metadata = {"render_fps": 30} 

    def __init__(self, grid_size=20, mode='shortest', render_mode=None):
        super(SmartLogisticsNavEnv, self).__init__()
        self.grid_size = grid_size
        self.mode = mode
        
        self.obstacles = [(i, i) for i in range(5, grid_size - 5)]
        
        self.action_space = spaces.Discrete(4)
        
        low = np.array([0, 0, 0, 0], dtype=np.int32)
        high = np.array([grid_size - 1, grid_size - 1, grid_size - 1, grid_size - 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.int32)
        
        self.target_pos = (grid_size - 1, grid_size - 1)
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos
        self.current_step = 0
        self.max_steps = grid_size * grid_size * 2
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return np.array([self.agent_pos[0], self.agent_pos[1], 
                         self.target_pos[0], self.target_pos[1]])

    def _calculate_reward(self, prev_dist):
        reward = 0
        
        current_dist = self._calculate_distance()
        distance_change = prev_dist - current_dist
        
        if self.mode == 'shortest':
            reward += distance_change * 10
            reward -= 0.1
        elif self.mode == 'fastest':
            reward += distance_change * 20
            reward -= 0.5
        elif self.mode == 'balanced':
            reward += distance_change * 5
            reward -= 0.2
        
        if self.agent_pos in self.obstacles:
            reward -= 1000

        if not (0 <= self.agent_pos[0] < self.grid_size and 0 <= self.agent_pos[1] < self.grid_size):
            reward -= 50

        if self.agent_pos == self.target_pos:
            reward += 10000
            
        return reward

    def _calculate_distance(self):
        return abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        prev_dist = self._calculate_distance()
        x, y = self.agent_pos
        
        if action == 0: y = min(y + 1, self.grid_size - 1)
        elif action == 1: y = max(y - 1, 0)
        elif action == 2: x = max(x - 1, 0)
        elif action == 3: x = min(x + 1, self.grid_size - 1)
        
        self.agent_pos = (x, y)
        self.current_step += 1
        
        reward = self._calculate_reward(prev_dist)
        
        terminated = self.agent_pos == self.target_pos
        truncated = self.current_step >= self.max_
