import matplotlib
matplotlib.use('Agg') # 解决 Streamlit Cloud 的无界面渲染错误

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import json, imageio, os
from datetime import datetime
import hashlib
import time # 用于 render 方法中的潜在优化

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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

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

    # 渲染方法 (用于 GIF 生成)
    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # 障碍物 (红色)
        for x, y in self.obstacles:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[y, x] = [255, 0, 0]
        
        # 目标 (绿色)
        gx, gy = self.target_pos
        grid[gy, gx] = [0, 255, 0]
        
        # Agent (蓝色)
        ax, ay = self.agent_pos
        if 0 <= ax < self.grid_size and 0 <= ay < self.grid_size:
             grid[ay, ax] = [0, 0, 255]
        
        # 使用 matplotlib 绘制网格
        fig, ax = plt.subplots(figsize=(self.grid_size/2, self.grid_size/2))
        ax.imshow(grid, origin='lower')
        ax.set_xticks(np.arange(self.grid_size))
        ax.set_yticks(np.arange(self.grid_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0) 
        
        # --- 图像生成修复点 ---
        fig.canvas.draw()
        
        # 尝试刷新事件，确保渲染完成（尤其在无界面环境中）
        try:
             fig.canvas.flush_events() 
        except NotImplementedError:
             pass
        
        # 获取 RGB 数据
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        
        # 确保尺寸正确 (height x width x 3)
        image = image.reshape(height, width, 3) 
        plt.close(fig) 
        # --- 修复点结束 ---
        
        return image

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
    
    # 保存模型 (可选)
    model.save(f"ppo_logistics_{mode}.zip")
    
    return model, logging_callback.logs

def run_test_and_render(model, mode, grid_size=20):
    """评估模型并生成导航 GIF"""
    print_status("Running Final Evaluation...")
    
    env = SmartLogisticsNavEnv(grid_size=grid_size, mode=mode)
    obs, _ = env.reset()
    
    images = []
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < env.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        # 记录图像帧
        images.append(env.render())
        
    env.close()
    
    # 保存 GIF
    gif_path = f"navigation_{mode}.gif"
    imageio.mimsave(gif_path, images, fps=10)
    
    test_result = {
        'steps': steps,
        'total_reward': total_reward,
        'reach_goal': env.agent_pos == env.target_pos
    }
    
    return test_result, gif_path

def save_pouw_to_blockchain(user_params, training_logs, test_result, model):
    """将 RL 训练结果作为 POUW 数据记录到区块链"""
    # 组合 POUW 数据
    pouw_data = {
        "user_params": user_params,
        "training_summary": {
            "start_time": str(datetime.now()),
            "total_timesteps": user_params['total_timesteps'],
            "final_reward": training_logs[-1]['avg_reward'] if training_logs else 0
        },
        "test_result": test_result,
        "model_architecture": str(model.policy.net)
    }
    
    # 获取最新区块
    latest_block = st.session_state.rl_pouw_chain.get_latest_block()
    
    # 创建新区块
    new_index = latest_block.index + 1
    new_block = Block(new_index, str(datetime.now()), pouw_data, latest_block.hash)
    
    # 挖矿并添加到链
    mined_hash = st.session_state.rl_pouw_chain.mine_block(new_block)
    
    # 验证链的有效性
    chain_valid = st.session_state.rl_pouw_chain.is_chain_valid()

    return {
        "block_index": new_block.index,
        "block_hash": mined_hash,
        "data": pouw_data,
        "is_chain_valid": chain_valid
    }

# ---------------------------------
# 5. Streamlit Web App Interface
# ---------------------------------

# 初始化 Streamlit 状态和区块链 (必须在所有 Streamlit 元素之前)
if 'rl_pouw_chain' not in st.session_state:
    st.session_state.rl_pouw_chain = SimpleBlockchain()

st.set_page_config(layout="wide")
st.title("🤖 RL-POUW 智能物流导航 MVP")
st.markdown("---")

# UI controls and inputs
col1, col2, col3 = st.columns(3)
with col1:
    selected_mode = st.selectbox(
        "选择导航模式:",
        ('shortest', 'fastest', 'balanced'),
        key='mode_select',
        help="Shortest: 优先最短路径; Fastest: 优先最快交付 (高时间惩罚); Balanced: 平衡路径和时间。"
    )
with col2:
    timesteps = st.number_input(
        "训练步数 (Timesteps):",
        min_value=10000,
        max_value=500000,
        value=150000,
        step=10000,
        key='timesteps_input',
        help="强化学习 Agent 的训练时长。步数越高，学习效果可能越好。"
    )
with col3:
    grid_size = st.number_input(
        "网格大小 (Grid Size):",
        min_value=10,
        max_value=30,
        value=20,
        step=5,
        key='grid_size_input',
        help="物流环境的网格地图尺寸 (N x N)."
    )


if st.button("🚀 开始训练 & 验证 POUW", use_container_width=True):
    # 确保在运行之前清理掉旧的 GIF，避免缓存问题
    if os.path.exists(f"navigation_{selected_mode}.gif"):
        os.remove(f"navigation_{selected_mode}.gif")
        
    st.markdown("### 训练日志")
    log_container = st.empty()
    
    with st.spinner(f"正在训练 {selected_mode.upper()} 模式... 请等待..."):
        # 训练 Agent
        with st.empty():
            model, training_logs = train_agent(selected_mode, timesteps, grid_size)

        # 运行评估
        # 错误发生在这里，但问题在 render 方法中
        test_result, gif_path = run_test_and_render(model, selected_mode, grid_size) 

        # 记录 POUW
        user_params = {"mode": selected_mode, "total_timesteps": timesteps, "grid_size": grid_size}
        pouw_record = save_pouw_to_blockchain(user_params, training_logs, test_result, model)
    
    # ---------------------------------
    # 结果展示
    # ---------------------------------
    st.success(f"训练和验证完成！模式: {selected_mode.upper()}")
    
    st.markdown("---")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.subheader("✅ 导航结果")
        st.metric("最终步数", test_result['steps'])
        st.metric("总奖励", f"{test_result['total_reward']:.2f}")
        st.metric("到达目标", "是" if test_result['reach_goal'] else "否")
        
        # 显示 GIF
        if os.path.exists(gif_path):
            st.image(gif_path, caption=f"物流 Agent 导航路径 ({selected_mode} 模式)")

    with col_res2:
        st.subheader("🔗 POUW 区块链记录")
        st.metric("新区块索引", pouw_record['block_index'])
        st.metric("链有效性", pouw_record['is_chain_valid'], 
                  delta="链验证失败，可能挖矿难度过高或数据校验问题" if not pouw_record['is_chain_valid'] else "链验证成功",
                  delta_color="inverse")
        st.write("区块哈希:", pouw_record['block_hash'][:12] + "...")
        with st.expander("查看完整的 POUW 数据"):
            st.json(pouw_record['data'])
            
    st.markdown("---")
    st.subheader("🌐 完整区块链状态")
    st.json([block.to_dict() for block in st.session_state.rl_pouw_chain.chain])
