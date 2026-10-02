import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.mot_dataset import MOT17Sequence
from src.tracking.rnn_tracker import RNNTracker
from src.metrics.tracker_metrics import EvaluatorMOT

def run_part2():
    seq_path = str(ROOT_DIR / "data/MOT17/train/MOT17-02-SDP")
    model_path = str(ROOT_DIR / "outputs/models/rnn_motion.pth")
    
    sequence = MOT17Sequence(seq_path)
    gt_data = sequence.load_gt()
    det_data = sequence.load_public_detections()
    
    # Inicia o rastreador da Parte 2 (RNN)
    tracker = RNNTracker(model_path, iou_threshold=0.3, max_age=5)
    pred_data = []
    
    frames = sorted(list(set([d['frame'] for d in det_data])))
    det_by_frame = {f: [d for d in det_data if d['frame'] == f] for f in frames}
    
    print("Iniciando inferência com RNN Tracker...")
    for f in frames:
        current_dets = det_by_frame.get(f, [])
        tracked_dets = tracker.update(current_dets, f)
        pred_data.extend(tracked_dets)
        
    evaluator = EvaluatorMOT(iou_threshold=0.5)
    metrics = evaluator.evaluate(gt_data, pred_data)
    
    print("=" * 50)
    print("RESULTADOS DA PARTE 2 - Trilha A (LSTM)")
    print(f"IDF1 Temporal:      {metrics['IDF1']:.4f}")
    print(f"ID Switches Total:  {metrics['IDSW']}")
    print(f"Fragmentações:      {metrics['Frag']}")
    print("=" * 50)

if __name__ == "__main__":
    run_part2()