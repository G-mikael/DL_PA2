import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.mot_dataset import MOT17Sequence
from src.synthetic.simulator import DetectorSimulator
from src.tracking.rnn_tracker import RNNTracker
from src.metrics.tracker_metrics import EvaluatorMOT
from src.detection.map_metric import compute_frame_ap

def run_stress_test():
    seq_path = str(ROOT_DIR / "data/MOT17/train/MOT17-02-SDP")
    model_path = str(ROOT_DIR / "outputs/models/rnn_motion.pth")
    
    sequence = MOT17Sequence(seq_path)
    gt_data = sequence.load_gt()
    original_dets = sequence.load_public_detections()
    
    # 3 Intensidades de Degradação (Leve, Média, Severa)
    intensities = [
        {'name': 'Original', 'drop': 0.0, 'noise': 0.0, 'fp': 0.0},
        {'name': 'Leve', 'drop': 0.1, 'noise': 5.0, 'fp': 0.02},
        {'name': 'Média', 'drop': 0.2, 'noise': 15.0, 'fp': 0.05},
        {'name': 'Severa', 'drop': 0.4, 'noise': 30.0, 'fp': 0.1}
    ]
    
    results_map = []
    results_idf1 = []
    
    print("Iniciando Teste de Estresse do Detector...")
    for config in intensities:
        print(f"\nTestando nível: {config['name']}")
        
        # 1. Estragar as detecções
        simulator = DetectorSimulator(drop_prob=config['drop'], noise_std=config['noise'], fp_rate=config['fp'])
        degraded_dets = simulator.corrupt(original_dets, img_size=(1920, 1080))
        
        frames = sorted(list(set([d['frame'] for d in gt_data])))
        det_by_frame = {f: [d for d in degraded_dets if d['frame'] == f] for f in frames}
        gt_by_frame = {f: [d for d in gt_data if d['frame'] == f] for f in frames}
        
        ap_per_frame = []
        pred_data = []
        tracker = RNNTracker(model_path, iou_threshold=0.3, max_age=5)
        
        # 2. Rastrear sobre os dados estragados
        for f in frames:
            current_dets = det_by_frame.get(f, [])
            current_gts = gt_by_frame.get(f, [])
            
            frame_ap = compute_frame_ap(current_gts, current_dets, iou_threshold=0.5)
            ap_per_frame.append(frame_ap)
            
            tracked_dets = tracker.update(current_dets, f)
            pred_data.extend(tracked_dets)
            
        # 3. Avaliar
        evaluator = EvaluatorMOT(iou_threshold=0.5)
        metrics = evaluator.evaluate(gt_data, pred_data)
        
        mAP_mean = np.mean(ap_per_frame)
        print(f"-> mAP: {mAP_mean:.4f} | IDF1: {metrics['IDF1']:.4f}")
        
        results_map.append(mAP_mean)
        results_idf1.append(metrics['IDF1'])

    # 4. Gráfico obrigatório
    labels = [c['name'] for c in intensities]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, results_map, width, label='mAP (Qualidade do Detector)', color='#D55E00')
    ax.bar(x + width/2, results_idf1, width, label='IDF1 (Desempenho Temporal)', color='#0072B2')

    ax.set_ylabel('Scores')
    ax.set_title('Parte 5: Absorção de Falhas do Detector pelo Modelo Temporal')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    output_dir = ROOT_DIR / "outputs" / "part5_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "estresse_detector.png", dpi=300)
    print("\n Gráfico salvo em outputs/part5_experiments/estresse_detector.png")

if __name__ == "__main__":
    run_stress_test()