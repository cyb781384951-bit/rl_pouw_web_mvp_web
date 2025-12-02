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

# ---------------------------------
# 1. 页面配置 (必须在所有其他 Streamlit 命令之前)
# ---------------------------------
st.set_page_config(
    page_title="RL-POUW 智能物流",
    layout="wide"
)

# ---------------------------------
# 辅助函数与类
# ---------------------------------
def print_status(message):
    """Prints status messages, checking if running in Streamlit."""
    if 'streamlit' in st.__doc__:
        st.info(message)
    else:
        print(message)

class LoggingCallback(BaseCallback):
    """
    Saves training metrics to a list for POUW inclusion.
    """
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.logs = []

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            avg_reward = np.mean(self.locals['rewards']) if len(self.locals['rewards']) > 0 else 0
            self.logs.append({
                'timesteps': self.num_timesteps,
                'avg_reward': float(f'{avg_reward:.2f}')
            })
        return True

# ---------------------------------
# 2. RL 环境定义
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
            reward += distance_change * 2
