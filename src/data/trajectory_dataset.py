import torch
from torch.utils.data import Dataset
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, gt_list, seq_len=10):
        self.seq_len = seq_len
        self.samples = []
        
        # Agrupar por ID
        trajectories = {}
        for row in gt_list:
            tid = row['id']
            if tid not in trajectories:
                trajectories[tid] = []
            # Normalização simples (dividindo por 1000) para estabilizar gradientes
            box = [row['bb_left']/1000.0, row['bb_top']/1000.0, 
                   row['bb_width']/1000.0, row['bb_height']/1000.0]
            trajectories[tid].append((row['frame'], box))
            
        # Ordenar por frame e extrair janelas deslizantes
        for tid, traj in trajectories.items():
            traj.sort(key=lambda x: x[0])
            boxes = [x[1] for x in traj]
            
            # Cria janelas de tamanho seq_len
            if len(boxes) > self.seq_len:
                for i in range(len(boxes) - self.seq_len):
                    seq = boxes[i : i + self.seq_len + 1] # +1 para ter o target final
                    self.samples.append(seq)
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        seq = torch.tensor(self.samples[idx], dtype=torch.float32)
        X = seq[:-1] # Inputs: instante t
        Y = seq[1:]  # Targets: instante t+1
        return X, Y