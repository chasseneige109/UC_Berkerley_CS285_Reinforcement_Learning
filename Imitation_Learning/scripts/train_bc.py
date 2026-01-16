import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

data = np.load("../data/expert_data.npz")
obs = torch.tensor(data["obs"], dtype = torch.float32)
acs = torch.tensor(data["acs"], dtype = torch.float32)

class StudentNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
            )
    
    def forward(self, x):
        return self.net(x)
    
input_dim = obs.shape[1]
output_dim = acs.shape[1]
student = StudentNet(input_dim, output_dim)
optimizer = optim.Adam(student.parameters(), lr = 1e-3)
criterion = nn.MSELoss() # Loss가 될 예정
batch_size = 64

for epoch in range(100):
    indices = torch.randperm(obs.size(0))
    epoch_loss = 0
    for i in range(0, obs.size(0), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_obs = obs[batch_idx]
        batch_acs = acs[batch_idx]
        
        #student 예측
        pred_acs = student(batch_obs)
        loss = criterion(pred_acs, batch_acs)
        
        #backprop and optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {epoch_loss/len(obs):.4f}")

os.makedirs("../models", exist_ok = True)
torch.save(student.state_dict(), "../models/student_bc_halfcheetah.pth")
print("완료")