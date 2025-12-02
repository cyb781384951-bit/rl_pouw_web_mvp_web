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
        return self._get_obs(), self._get_info()

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
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_info(self):
        return {"distance": self._calculate_distance(), "agent_pos": self.agent_pos}

    def render(self):
        return None 
        
# ---------------------------------
# 3. POUW 区块链逻辑 (已修复语法错误)
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
        self.difficulty = 4

    def create_genesis_block(self):
        return Block(0, str(datetime.now()), 
                     {"message": "Genesis Block for RL POUW Chain"}, "0")

    def get_latest_block(self):
        return self.chain[-1]

    def mine_block(self, new_block):
        target = '0' * self.difficulty
        while new_block.hash[:self.difficulty] != target:
            new_block.nonce += 1
            new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)
        return new_block.hash

    # --- 修复点：完整的函数定义 ---
    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            if current_block.hash != current_block.calculate_hash():
                return False
            
            if current_block.previous_hash != previous_block.hash:
                return False
                
            target = '0' * self.difficulty
            if current_block.hash[:self.difficulty] != target:
                return False     
        return True

# ---------------------------------
# 4. 核心训练和评估函数
# ---------------------------------
def train_agent(mode, timesteps, grid_size=20):
    print_status(f"Training PPO Agent for {mode} mode with {timesteps} steps...")
    env = SmartLogisticsNavEnv(grid_size=grid_size, mode=mode)
    vec_env = make_vec_env(lambda: env, n_envs=1)
    logging_callback = LoggingCallback(verbose=1)
    model = PPO("MlpPolicy", vec_env, verbose=0, device="auto", tensorboard_log=None)
    model.learn(total_timesteps=timesteps, callback=logging_callback)
    return model, logging_callback.logs

def run_test_and_render(model, mode, grid_size=20):
    print_status("Running Final Evaluation (GIF rendering skipped)...")
    env = SmartLogisticsNavEnv(grid_size=grid_size, mode=mode)
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < env.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    env.close()
    
    test_result = {
        'steps': steps,
        'total_reward': total_reward,
        'reach_goal': env.agent_pos == env.target_pos
    }
    return test_result, "navigation_skipped.gif"

def save_pouw_to_blockchain(user_params, training_logs, test_result, model):
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
    latest_block = st.session_state.rl_pouw_chain.get_latest_block()
    new_index = latest_block.index + 1
    new_block = Block(new_index, str(datetime.now()), pouw_data, latest_block.hash)
    mined_hash = st.session_state.rl_pouw_chain.mine_block(new_block)
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

# 初始化状态 (必须在 set_page_config 之后)
if 'rl_pouw_chain' not in st.session_state:
    st.session_state.rl_pouw_chain = SimpleBlockchain()

st.title("🤖 RL-POUW 智能物流导航 MVP")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    selected_mode = st.selectbox("选择导航模式:", ('shortest', 'fastest', 'balanced'))
with col2:
    timesteps = st.number_input("训练步数 (Timesteps):", value=150000, step=10000)
with col3:
    grid_size = st.number_input("网格大小 (Grid Size):", value=20, step=5)

if st.button("🚀 开始训练 & 验证 POUW", use_container_width=True):
    st.markdown("### 训练日志")
    log_container = st.empty()
    
    with st.spinner(f"正在训练 {selected_mode.upper()} 模式... 请等待..."):
        model, training_logs = train_agent(selected_mode, timesteps, grid_size)
        test_result, gif_path = run_test_and_render(model, selected_mode, grid_size) 
        user_params = {"mode": selected_mode, "total_timesteps": timesteps, "grid_size": grid_size}
        pouw_record = save_pouw_to_blockchain(user_params, training_logs, test_result, model)
    
    st.success(f"训练和验证完成！模式: {selected_mode.upper()}")
    st.markdown("---")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.subheader("✅ 导航结果")
        st.metric("最终步数", test_result['steps'])
        st.metric("总奖励", f"{test_result['total_reward']:.2f}")
        st.metric("到达目标", "是" if test_result['reach_goal'] else "否")
        st.info("⚠️ 注意：为了保证服务器稳定性，已禁用在线 GIF 渲染。")

    with col_res2:
        st.subheader("🔗 POUW 区块链记录")
        st.metric("新区块索引", pouw_record['block_index'])
        st.metric("链有效性", pouw_record['is_chain_valid'], 
                  delta="验证失败" if not pouw_record['is_chain_valid'] else "验证成功",
                  delta_color="inverse")
        st.write("区块哈希:", pouw_record['block_hash'][:12] + "...")
        with st.expander("查看完整数据"):
            st.json(pouw_record['data'])
            
    st.markdown("---")
    st.subheader("🌐 区块链总览")
    st.json([block.to_dict() for block in st.session_state.rl_pouw_chain.chain])
