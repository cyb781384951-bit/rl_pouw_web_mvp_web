import streamlit as st
import numpy as np
# 移除 Matplotlib 导入和 use('Agg') 设置，因为我们不再使用它
# import matplotlib.pyplot as plt 
# import matplotlib
# matplotlib.use('Agg')

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import json, imageio, os
from datetime import datetime
import hashlib
import time

# ---------------------------------
# Streamlit/Colab 兼容的打印函数 (替代 st.write)
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
    # 移除 render_modes metadata
    metadata = {"render_fps": 30} 

    def __init__(self, grid_size=20, mode='shortest', render_mode=None):
        super(SmartLogisticsNavEnv, self).__init__()
        self.grid_size = grid_size
        self.mode = mode
        
        # 障碍物位置 (示例：一条对角线障碍)
        self.obstacles = [(i, i) for i in range(5, grid_size - 5)]
        
        # 动作空间：0: 上, 1: 下, 2: 左, 3: 右
        self.action_space = spaces.Discrete(4)
        
        # 观测空间：(agent_x, agent_y, target_x, target_y)
        low = np.array([0, 0, 0, 0], dtype=np.int32)
        high = np.array([grid_size - 1, grid_size - 1, grid_size - 1, grid_size - 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.int32)
        
        # 目标位置 (右下角) 和起始位置 (左上角)
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
        
        # 1. 距离奖励 (Progress)
        current_dist = self._calculate_distance()
        distance_change = prev_dist - current_dist
        
        # 2. 模式特定的奖励/惩罚
        if self.mode == 'shortest':
            # 鼓励靠近目标
            reward += distance_change * 10
            # 小惩罚时间步长，鼓励快速结束
            reward -= 0.1
        elif self.mode == 'fastest':
            # 鼓励快速靠近 (更高的距离奖励)
            reward += distance_change * 20
            # 较高惩罚时间步长
            reward -= 0.5
        elif self.mode == 'balanced':
            # 均衡奖励 (距离和时间)
            reward += distance_change * 5
            reward -= 0.2
        
        # 3. 碰撞惩罚 (Punishment)
        if self.agent_pos in self.obstacles:
            reward -= 1000  # 极高的碰撞惩罚

        # 4. 边界惩罚
        if not (0 <= self.agent_pos[0] < self.grid_size and 0 <= self.agent_pos[1] < self.grid_size):
            reward -= 50  # 边界惩罚 (RL Agent 不应该越界)

        # 5. 胜利奖励 (Goal)
        if self.agent_pos == self.target_pos:
            reward += 10000  # 极高的胜利奖励
            
        return reward

    def _calculate_distance(self):
        # 曼哈顿距离
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
        
        # 移动 (0:上, 1:下, 2:左, 3:右)
        if action == 0: y = min(y + 1, self.grid_size - 1)
        elif action == 1: y = max(y - 1, 0)
        elif action == 2: x = max(x - 1, 0)
        elif action == 3: x = min(x + 1, self.grid_size - 1)
        
        self.agent_pos = (x, y)
        self.current_step += 1
        
        reward = self._calculate_reward(prev_dist)
        
        terminated = self.agent_pos == self.target_pos
        truncated = self.current_step >= self.max_steps

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _get_info(self):
        return {"distance": self._calculate_distance(), "agent_pos": self.agent_pos}

    # 渲染方法 (改为返回 None，以避免错误)
    def render(self):
        # 彻底移除 Matplotlib 绘图代码
        return None 
        
# ---------------------------------
# 3. POUW 区块链逻辑 (Block & SimpleBlockchain)
# ---------------------------------
class Block:
    def __init__(self, index, timestamp, data, previous_hash='0'):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }

class SimpleBlockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 4 # 挖矿难度：哈希必须以 4 个零开头

    def create_genesis_block(self):
        # 创世区块使用固定的 POUW 数据
        return Block(0, str(datetime.now()), 
                     {"message": "Genesis Block for RL POUW Chain"}, "0")

    def get_latest_block(self):
        return self.chain[-1]

    def mine_block(self, new_block):
        # 找到满足难度要求的 Nonce
        target = '0' * self.difficulty
        while new_block.hash[:self.difficulty] != target:
            new_block.nonce += 1
            new_block.hash = new_block.calculate_hash()
        
        # 将挖出的区块添加到链中
        self.chain.append(new_block)
        return new_block.hash

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # 1. 检查哈希是否正确计算 (防止数据被篡改)
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # 2. 检查链的连接是否正确 (防止区块顺序被篡改)
            if current_block.previous_hash != previous_block.hash:
                return False
                
            # 3. 检查哈希是否满足挖矿难度
            target = '0' * self.difficulty
            if current_block.hash[:self.difficulty] != target:
                return False
                
        return True

# ---------------------------------
# 4. 核心训练和评估函数
# ---------------------------------

def train_agent(mode, timesteps, grid_size=20):
    """训练 PPO Agent 并返回模型和日志"""
    print_status(f"Training PPO Agent for {mode} mode with {timesteps} steps...")
    
    # 创建环境
    env = SmartLogisticsNavEnv(grid_size=grid_size, mode=mode)
    vec_env = make_vec_env(lambda: env, n_envs=1)
    
    # 创建日志回调
    logging_callback = LoggingCallback(verbose=1)

    # 初始化 PPO 模型
    model = PPO("MlpPolicy", vec_env, verbose=0, device="auto", tensorboard_log=None)
    
    # 训练
    model.learn(total_timesteps=timesteps, callback=logging_callback)
    
    # 移除模型保存
    # model.save(f"ppo_logistics_{mode}.zip") 
    
    return model, logging_callback.logs

def run_test_and_render(model, mode, grid_size=20):
    """评估模型，跳过 GIF 渲染以避免服务器错误"""
    print_status("Running Final Evaluation (GIF rendering skipped)...")
    
    env = SmartLogisticsNavEnv(grid_size=grid_size, mode=mode)
    obs, _ = env.reset()
    
    # 移除 images 列表
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < env.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        # 移除 env.render() 调用
        # env.render() 
        
    env.close()
    
    # 移除 GIF 保存
    gif_path = "navigation_skipped.gif" # 返回一个虚拟路径
    
    test
