import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.mot_dataset import MOT17Sequence
from src.data.trajectory_dataset import TrajectoryDataset
from src.models.rnn_ablation import AblationMotionModel
from src.tracking.ablation_tracker import AblationTracker
from src.metrics.tracker_metrics import EvaluatorMOT

def run_ablation():
    seq_path = str(ROOT_DIR / "data/MOT17/train/MOT17-02-SDP")
    seq = MOT17Sequence(seq_path)
    gt_data = seq.load_gt()
    det_data = seq.load_public_detections()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cells = ['RNN', 'GRU', 'LSTM']
    windows = [4, 8, 16, 32]
    seeds = [2026, 2027, 2028]
    
    results = {cell: {w: [] for w in windows} for cell in cells}
    output_dir = ROOT_DIR / "outputs" / "part3_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames = sorted(list(set([d['frame'] for d in det_data])))
    det_by_frame = {f: [d for d in det_data if d['frame'] == f] for f in frames}
    
    for cell in cells:
        for T in windows:
            print(f"\n--- Iniciando configuração: Célula {cell} | Janela BPTT {T} ---")
            dataset = TrajectoryDataset(gt_data, seq_len=T)
            
            for s_idx, seed in enumerate(seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                model = AblationMotionModel(cell_type=cell).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.SmoothL1Loss()
                
                # Treino reduzido (10 épocas são suficientes para comparar a estabilidade)
                model.train()
                for epoch in range(10):
                    for X, Y in dataloader:
                        X, Y = X.to(device), Y.to(device)
                        optimizer.zero_grad()
                        preds, _ = model(X)
                        loss = criterion(preds, Y)
                        loss.backward()
                        optimizer.step()
                
                model_path = output_dir / f"{cell}_T{T}_seed{seed}.pth"
                torch.save(model.state_dict(), model_path)
                
                # Avaliação (Tracker)
                tracker = AblationTracker(str(model_path), cell_type=cell, iou_threshold=0.3, max_age=5)
                pred_data = []
                for f in frames:
                    current_dets = det_by_frame.get(f, [])
                    tracked_dets = tracker.update(current_dets, f)
                    pred_data.extend(tracked_dets)
                    
                evaluator = EvaluatorMOT(iou_threshold=0.5)
                metrics = evaluator.evaluate(gt_data, pred_data)
                
                print(f"Seed {seed} -> IDF1: {metrics['IDF1']:.4f}")
                results[cell][T].append(metrics['IDF1'])

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n Ablação concluída. Resultados salvos.")

if __name__ == "__main__":
    run_ablation()