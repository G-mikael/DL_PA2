import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.mot_dataset import MOT17Sequence
from src.data.trajectory_dataset import TrajectoryDataset
from src.models.rnn_motion import RNNMotionModel

def train():
    # 1. Carregar dados
    seq = MOT17Sequence(str(ROOT_DIR / "data/MOT17/train/MOT17-02-SDP"))
    gt_data = seq.load_gt()
    
    dataset = TrajectoryDataset(gt_data, seq_len=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Configurar Modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNMotionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss()
    
    print(f"Iniciando treinamento no {device} com {len(dataset)} sequencias...")
    
    # 3. Loop de Treino
    model.train()
    for epoch in range(1, 21):  # 20 épocas
        total_loss = 0
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            
            preds, _ = model(X)
            loss = criterion(preds, Y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoca {epoch:02d} | Perda Media: {total_loss/len(dataloader):.6f}")
        
    # 4. Salvar pesos
    save_dir = ROOT_DIR / "outputs" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "rnn_motion.pth")
    print("Pesos salvos em outputs/models/rnn_motion.pth!")

if __name__ == "__main__":
    train()