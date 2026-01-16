# Written by Gemini

import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.utils.save_video import save_video
import os

# 폴더 생성 확인
os.makedirs("../videos", exist_ok=True)

# 1. 환경 생성
env = gym.make("HalfCheetah-v4", render_mode="rgb_array_list")

# 2. 모델 로드
model = PPO.load("../models/expert_halfcheetah")

# 3. 주행 테스트
obs, _ = env.reset()
for _ in range(300): # 테스트를 위해 300스텝만 짧게 찍어봅시다
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

# 4. 영상 저장
print("영상 인코딩 시작... (잠시 기다려주세요)")
save_video(
    env.render(),
    "../videos",
    fps=30,
    name_prefix="halfcheetah_test"
)

env.close() # 환경을 닫아줘야 파일이 안전하게 저장됩니다!
print("저장 완료!")