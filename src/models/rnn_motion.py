import torch
import torch.nn as nn

class RNNMotionModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, 4)
        out, hidden = self.lstm(x, hidden)
        pred = self.fc(out) # shape: (batch, seq_len, 4)
        return pred, hidden