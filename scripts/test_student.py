
import gymnasium as gym
import torch
import torch.nn as nn
import imageio
import os

class StudentNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# 모델
MODEL_PATH = "../models/student_bc_halfcheetah.pth"
student = StudentNet(17, 6)
student.load_state_dict(torch.load(MODEL_PATH))
student.eval()

# 환경 설정 
env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
obs, _ = env.reset()

frames = [] 
print("--- 제자 로봇 주행 테스트 시작 ---")

# 주행 및 프레임 수집
for i in range(300): # 300프레임 약 10초
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = student(obs_t).squeeze(0).numpy()
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    frames.append(env.render()) 
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()

# 강제 인코딩 및 저장
video_dir = "../videos"
os.makedirs(video_dir, exist_ok=True)
video_path = os.path.join(video_dir, "student_final_success.mp4")

if len(frames) > 0:
    print(f"인코딩 시작... {len(frames)} 프레임을 영상으로 변환합니다.")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"--- 저장 완료! ---")
    print(f"파일 위치: {os.path.abspath(video_path)}")
else:
    print("에러: 수집된 프레임이 없습니다. 환경 설정을 확인하세요.")
