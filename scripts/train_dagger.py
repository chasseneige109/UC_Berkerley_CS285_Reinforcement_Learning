# DAgger
# new_obs by student -> predict expert -> relearning

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import os

class StudentNet(nn.Module):
    def __init__ (self, input_dim, output_dim):
        super(StudentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
env = gym.make("HalfCheetah-v4")
expert = PPO.load("../models/expert_halfcheetah")
student = StudentNet(17, 6)
student.load_state_dict(torch.load("../models/student_bc_halfcheetah.pth"))

optimizer = optim.Adam(student.parameters(), lr=1e-3)
criterion = nn.MSELoss()
batch_size = 64

# expert data
data = np.load("../data/expert_data.npz")
all_obs = data["obs"]
all_acs = data["acs"]

# DAgger dataset 모으기(확장)

for iter in range(5):
    print(f"--- DAgger iter {iter + 1}---")
    
    new_obs, new_acs = [], []
    obs, _ = env.reset()
    for _ in range(2000):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): # new_obs만 얻으면 되니 기록 안함
            action = student(obs_t).squeeze(0).numpy()
            
        expert_action, _ = expert.predict(obs, deterministic=True)
        new_obs.append(obs)
        new_acs.append(expert_action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    all_obs = np.concatenate([all_obs, np.array(new_obs)] , axis=0)
    all_acs =  np.concatenate([all_acs, np.array(new_acs)] , axis=0)


# DAgger로 확장된 데이터로 student 재학습
obs_tensor = torch.tensor(all_obs, dtype=torch.float32)
acs_tensor = torch.tensor(all_acs, dtype=torch.float32)

for epoch in range(50):
    indices = torch.randperm(obs_tensor.size(0))
    epoch_loss = 0
    for i in range(0, len(all_obs), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_obs = obs_tensor[batch_idx]
        batch_acs = acs_tensor[batch_idx]
        
        pred = student(batch_obs)
        loss = criterion(pred, batch_acs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
print(" 학습 끗 ")
    

os.makedirs("../models", exist_ok=True)
torch.save(student.state_dict(), "../models/student_dagger_halfcheetah.pth")
print("저장까지 완전 끗")