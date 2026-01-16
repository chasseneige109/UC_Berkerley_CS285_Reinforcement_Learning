import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os

# 경로
DATA_DIR = "../data"
MODEL_DIR = "../models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# env 생성
env = gym.make("HalfCheetah-v4")

# expert
# 5만 step해보니까 개똥멍청이임
# 50만 step하니까 뒤집혀서 발광함
print("--- expert 학습시작 ---")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)

# 모델 저장하기
model.save(f"{MODEL_DIR}/expert_halfcheetah")
print(f"모델 저장 완료: {MODEL_DIR}/expert_halfcheetah.zip")

# expert Data 수집 (Imitation Learning용)
print("--- 데이터 수집 중... ---")
observations, actions = [], []
obs, _ = env.reset()

for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    observations.append(obs)
    actions.append(action)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

# 데이터셋 저장
np.savez(f"{DATA_DIR}/expert_data.npz", obs=np.array(observations), acs=np.array(actions))
print("완료")