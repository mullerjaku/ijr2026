import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.ln1(x)
        out = self.fc1(out)
        out = self.act(out)
        
        out = self.ln2(out)
        out = self.fc2(out)
        
        return residual + out

class WorldModel(nn.Module):
    def __init__(self, state_dim=3, action_dim=3, hidden_dim=128):
        super().__init__()
        self.input = nn.Linear(state_dim + action_dim, hidden_dim)
        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)

        self.ln_out = nn.LayerNorm(hidden_dim) 
        self.output = nn.Linear(hidden_dim, state_dim)

        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        h = torch.relu(self.input(x))
        
        h = self.block1(h)
        h = self.block2(h)
        
        h = self.ln_out(h)
        delta_state = torch.tanh(self.output(h)) * self.output_scale
        next_state = state + delta_state
        return torch.clamp(next_state, 0.0, 1.0)

 
