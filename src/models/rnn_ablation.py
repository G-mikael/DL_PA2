import torch
import torch.nn as nn

class AblationMotionModel(nn.Module):
    def __init__(self, input_dim=4, base_hidden=64, cell_type='LSTM'):
        super().__init__()
        self.cell_type = cell_type
        
        # Ajuste do hidden_dim para manter o orçamento de parâmetros similar (~17.000)
        # LSTM base: 4 * (4*64 + 64*64) ~ 17k
        # GRU: 3 * (4*H + H*H) -> H ~ 74
        # RNN: 1 * (4*H + H*H) -> H ~ 128
        if cell_type == 'LSTM':
            self.hidden_dim = base_hidden
            self.rnn = nn.LSTM(input_dim, self.hidden_dim, batch_first=True)
        elif cell_type == 'GRU':
            self.hidden_dim = int(base_hidden * 1.15)
            self.rnn = nn.GRU(input_dim, self.hidden_dim, batch_first=True)
        elif cell_type == 'RNN':
            self.hidden_dim = int(base_hidden * 2.0)
            self.rnn = nn.RNN(input_dim, self.hidden_dim, batch_first=True)
        else:
            raise ValueError("Tipo de célula inválido.")
            
        self.fc = nn.Linear(self.hidden_dim, input_dim)
        
    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        pred = self.fc(out)
        return pred, hidden