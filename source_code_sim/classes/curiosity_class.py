import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomTargetNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.fc(x)

class PredictorNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class RNDModule:
    def __init__(self, input_size, output_size=128, device="cpu"):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size

        self.target_net = RandomTargetNetwork(input_size, output_size).to(device)
        self.predictor_net = PredictorNetwork(input_size, output_size).to(device)

        self.predictor_optimizer = torch.optim.Adam(self.predictor_net.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def compute_curiosity(self, state_np):
        """
        Compute curiosity reward from a numpy array input.
        """
        state_tensor = torch.tensor(state_np, dtype=torch.float32).to(self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            target_output = self.target_net(state_tensor)

        predicted_output = self.predictor_net(state_tensor)
        curiosity = ((predicted_output - target_output) ** 2).mean(dim=1)
        return curiosity.detach().cpu().numpy()

    def train_step(self, state_np):
        """
        Train the predictor network to minimize prediction error.
        """
        self.predictor_net.train()
        state_tensor = torch.tensor(state_np, dtype=torch.float32).to(self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            target_output = self.target_net(state_tensor)

        predicted_output = self.predictor_net(state_tensor)
        loss = self.criterion(predicted_output, target_output)

        self.predictor_optimizer.zero_grad()
        loss.backward()
        self.predictor_optimizer.step()

        return loss.item()
