import torch
import torch.nn as nn
import torch.nn.functional as F

class ChaturajiNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 8x8x40 (channels last)
        self.conv1 = nn.Conv2d(40, 128, 3, padding=1)  # Reduced from 256
        self.bn1 = nn.BatchNorm2d(128)  # Was 256
        self.resblocks = nn.ModuleList([ResBlock(128) for _ in range(3)])  # Fewer blocks
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, 1)  # Was 256
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*8*8, 4096)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 1, 1)  # Was 256
        self.value_bn = nn.BatchNorm2d(1)
        # self.value_fc1 = nn.Linear(8*8, 256)
        # self.value_fc2 = nn.Linear(256, 1)
        self.value_fc1 = nn.Linear(8*8, 128)  # Reduced - not strictly needed but just for speedup
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.resblocks:
            x = block(x)
        
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        
        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)